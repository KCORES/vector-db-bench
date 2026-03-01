#!/usr/bin/env python3
"""
Generate ground truth for SIFT1M dataset via brute-force L2 nearest neighbor search.

Reads base vectors and query vectors from fvecs files, computes exact L2 distances
between each query and all base vectors, and outputs the Top-K nearest neighbor IDs.

Supports chunked processing with intermediate results saved to disk, enabling:
- Resume from interruption (already-computed chunks are skipped)
- Lower memory usage (each chunk is written independently)

Usage:
    python scripts/generate_ground_truth.py [--data-dir data/] [--top-k 100] [--workers 1] [--chunk-size 1000]
"""

import argparse
import glob
import json
import math
import os
import struct
import sys
import time
from multiprocessing import Pool


def read_fvecs(filepath):
    """Parse an fvecs file and return a list of float32 vectors.

    Each vector is stored as:
        4 bytes (int32 dimension) + dim * 4 bytes (float32 values)

    Args:
        filepath: Path to the .fvecs file.

    Returns:
        List of lists of floats.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid or dimensions are inconsistent.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"fvecs file not found: {filepath}")

    vectors = []
    expected_dim = None
    file_size = os.path.getsize(filepath)

    with open(filepath, "rb") as f:
        while f.tell() < file_size:
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                if len(dim_bytes) == 0:
                    break
                raise ValueError(
                    f"Unexpected end of file reading dimension at byte {f.tell() - len(dim_bytes)} "
                    f"in {filepath}"
                )

            (dim,) = struct.unpack("<i", dim_bytes)
            if dim <= 0:
                raise ValueError(
                    f"Invalid dimension {dim} at vector index {len(vectors)} in {filepath}"
                )

            if expected_dim is None:
                expected_dim = dim
            elif dim != expected_dim:
                raise ValueError(
                    f"Dimension mismatch at vector index {len(vectors)}: "
                    f"expected {expected_dim}, got {dim} in {filepath}"
                )

            vec_bytes = f.read(dim * 4)
            if len(vec_bytes) < dim * 4:
                raise ValueError(
                    f"Unexpected end of file reading vector data at vector index {len(vectors)} "
                    f"in {filepath}. Expected {dim * 4} bytes, got {len(vec_bytes)}"
                )

            values = list(struct.unpack(f"<{dim}f", vec_bytes))
            vectors.append(values)

    if not vectors:
        raise ValueError(f"No vectors found in {filepath}")

    return vectors


def _l2_distance_squared_python(a, b):
    """Compute squared L2 distance between two vectors (pure Python).

    We skip the sqrt since it's monotonic — sorting by squared distance
    gives the same ordering as sorting by actual distance.
    """
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


def _knn_single_query_numpy(base_vectors, query_vec, top_k):
    """Compute KNN for a single query vector using numpy.

    Returns:
        List of neighbor IDs (length top_k).
    """
    import numpy as np

    diff = base_vectors - query_vec
    dists = np.sum(diff * diff, axis=1)
    if top_k < len(dists):
        top_indices = np.argpartition(dists, top_k)[:top_k]
        sorted_order = np.argsort(dists[top_indices])
        return top_indices[sorted_order].tolist()
    else:
        return np.argsort(dists)[:top_k].tolist()


def _knn_single_query_python(base_vectors, query_vec, top_k):
    """Compute KNN for a single query vector using pure Python.

    Returns:
        List of neighbor IDs (length top_k).
    """
    import heapq

    dists = []
    for j, base in enumerate(base_vectors):
        d = _l2_distance_squared_python(query_vec, base)
        dists.append((d, j))
    top = heapq.nsmallest(top_k, dists, key=lambda x: x[0])
    return [idx for _, idx in top]


def _process_chunk_to_disk(args):
    """Process a chunk of queries and write results directly to a JSON file.

    This is the worker function for both single-process and multiprocessing modes.
    Each chunk is independently saved to disk, enabling resume and lower memory usage.

    Args:
        args: Tuple of (base_data, query_chunk, top_k, chunk_id,
               query_id_offset, output_path, use_numpy).

    Returns:
        output_path on success.
    """
    base_data, query_chunk, top_k, chunk_id, query_id_offset, output_path, use_numpy = args

    if use_numpy:
        import numpy as np
        if not isinstance(base_data, np.ndarray):
            base_data = np.array(base_data, dtype=np.float32)
        if not isinstance(query_chunk, np.ndarray):
            query_chunk = np.array(query_chunk, dtype=np.float32)
        num_queries = query_chunk.shape[0]
    else:
        num_queries = len(query_chunk)

    results = []
    start_time = time.time()

    for i in range(num_queries):
        if use_numpy:
            neighbors = _knn_single_query_numpy(base_data, query_chunk[i], top_k)
        else:
            neighbors = _knn_single_query_python(base_data, query_chunk[i], top_k)

        results.append({
            "query_id": query_id_offset + i,
            "neighbors": neighbors,
        })

        if (i + 1) % 100 == 0 or i == num_queries - 1:
            elapsed = time.time() - start_time
            qps = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (num_queries - i - 1) / qps if qps > 0 else 0
            print(
                f"  [Chunk {chunk_id}] {i + 1}/{num_queries} queries "
                f"({elapsed:.1f}s elapsed, {eta:.1f}s ETA)",
                flush=True,
            )

    # Write chunk to disk
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    return output_path


def _merge_chunk_files(chunk_paths, output_path, total_queries):
    """Merge sorted chunk JSON files into the final ground_truth.json.

    Reads each chunk file sequentially and streams into the output file,
    so only one chunk is in memory at a time.

    Args:
        chunk_paths: List of chunk file paths in order.
        output_path: Final output JSON path.
        total_queries: Expected total number of query results.
    """
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("[")
        first = True
        count = 0
        for cp in chunk_paths:
            with open(cp, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
            for entry in chunk_data:
                if not first:
                    out.write(",")
                json.dump(entry, out)
                first = False
                count += 1
        out.write("]")

    if count != total_queries:
        print(
            f"Warning: merged {count} entries but expected {total_queries}",
            file=sys.stderr,
        )


def _cleanup_chunk_files(chunk_paths):
    """Remove temporary chunk files after successful merge."""
    for cp in chunk_paths:
        try:
            os.remove(cp)
        except OSError:
            pass


def compute_ground_truth(base_vectors, query_vectors, top_k=100, workers=1):
    """Compute ground truth nearest neighbors via brute-force search.

    This is the in-memory API used by tests and programmatic callers.
    For large datasets, use main() which does chunked-to-disk processing.

    Attempts to use numpy for performance. Falls back to pure Python if
    numpy is not available.

    Args:
        base_vectors: List of float vectors (base dataset).
        query_vectors: List of float vectors (queries).
        top_k: Number of nearest neighbors per query.
        workers: Number of parallel worker processes (default: 1, single-threaded).

    Returns:
        List of dicts: [{"query_id": int, "neighbors": [int, ...]}, ...]
    """
    use_numpy = False
    try:
        import numpy as np
        use_numpy = True
        print("  Using numpy for brute-force computation")
        base_data = np.array(base_vectors, dtype=np.float32)
        query_data = np.array(query_vectors, dtype=np.float32)
    except ImportError:
        print("  numpy not available, using pure Python (this will be slow)")
        base_data = base_vectors
        query_data = query_vectors

    if workers > 1:
        print(f"  Parallel mode: {workers} workers")

    num_queries = len(query_vectors)

    if workers <= 1:
        # Single-process in-memory path
        results = []
        start_time = time.time()
        for i in range(num_queries):
            if use_numpy:
                neighbors = _knn_single_query_numpy(base_data, query_data[i], top_k)
            else:
                neighbors = _knn_single_query_python(base_data, query_data[i], top_k)
            results.append({"query_id": i, "neighbors": neighbors})

            if (i + 1) % 100 == 0 or i == num_queries - 1:
                elapsed = time.time() - start_time
                qps = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (num_queries - i - 1) / qps if qps > 0 else 0
                print(
                    f"  Progress: {i + 1}/{num_queries} queries "
                    f"({elapsed:.1f}s elapsed, {eta:.1f}s ETA)",
                    flush=True,
                )
        return results

    # Multi-process: use temp files, then merge in memory
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="gt_chunks_")
    chunk_size = math.ceil(num_queries / workers)

    # Convert to lists for pickling
    if use_numpy:
        base_list = base_data.tolist()
        query_list = query_data.tolist()
    else:
        base_list = base_data
        query_list = query_data

    chunks = []
    for idx in range(workers):
        start = idx * chunk_size
        end = min(start + chunk_size, num_queries)
        if start >= num_queries:
            break
        chunk_path = os.path.join(tmpdir, f"chunk_{idx}.json")
        chunks.append((base_list, query_list[start:end], top_k, idx, start, chunk_path, use_numpy))

    print(f"  Dispatching {len(chunks)} worker processes...")
    with Pool(processes=len(chunks)) as pool:
        pool.map(_process_chunk_to_disk, chunks)

    # Merge from disk
    chunk_paths = [c[5] for c in chunks]
    results = []
    for cp in chunk_paths:
        with open(cp, "r", encoding="utf-8") as f:
            results.extend(json.load(f))

    _cleanup_chunk_files(chunk_paths)
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    return results


def _chunk_path(data_dir, chunk_id):
    """Return the path for a ground truth chunk file."""
    return os.path.join(data_dir, f"ground_truth_chunk_{chunk_id}.json")


def _is_valid_chunk(existing_chunk, q_start, q_end, top_k):
    """Validate whether a previously saved chunk can be reused safely."""
    expected_len = q_end - q_start
    if not isinstance(existing_chunk, list) or len(existing_chunk) != expected_len:
        return False

    for offset, row in enumerate(existing_chunk):
        if not isinstance(row, dict):
            return False
        expected_query_id = q_start + offset
        if row.get("query_id") != expected_query_id:
            return False
        neighbors = row.get("neighbors")
        if not isinstance(neighbors, list) or len(neighbors) != top_k:
            return False
        if not all(isinstance(n, int) for n in neighbors):
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth via brute-force L2 nearest neighbor search."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        help="Directory containing fvecs files and for JSON output (default: data/)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of nearest neighbors per query (default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1, single-threaded)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of queries per chunk for disk checkpointing (default: 1000)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    top_k = args.top_k
    workers = args.workers
    chunk_size = args.chunk_size

    if top_k <= 0:
        print("Error: --top-k must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    if workers <= 0:
        print("Error: --workers must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    if chunk_size <= 0:
        print("Error: --chunk-size must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    base_fvecs = os.path.join(data_dir, "sift_base.fvecs")
    query_fvecs = os.path.join(data_dir, "sift_query.fvecs")
    output_path = os.path.join(data_dir, "ground_truth.json")

    # Read base vectors
    print(f"Reading base vectors from {base_fvecs}...")
    try:
        base_vectors = read_fvecs(base_fvecs)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(base_vectors)} vectors, dim={len(base_vectors[0])}")

    # Read query vectors
    print(f"Reading query vectors from {query_fvecs}...")
    try:
        query_vectors = read_fvecs(query_fvecs)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(query_vectors)} vectors, dim={len(query_vectors[0])}")

    if top_k > len(base_vectors):
        print(
            f"Warning: top_k ({top_k}) exceeds number of base vectors ({len(base_vectors)}). "
            f"Clamping to {len(base_vectors)}.",
            file=sys.stderr,
        )
        top_k = len(base_vectors)

    num_queries = len(query_vectors)
    num_chunks = math.ceil(num_queries / chunk_size)

    # Detect numpy
    use_numpy = False
    try:
        import numpy as np
        use_numpy = True
        print("  Using numpy for brute-force computation")
        base_data = np.array(base_vectors, dtype=np.float32)
        query_data = np.array(query_vectors, dtype=np.float32)
    except ImportError:
        print("  numpy not available, using pure Python (this will be slow)")
        base_data = base_vectors
        query_data = query_vectors

    # Free original lists if numpy took over
    if use_numpy:
        del base_vectors, query_vectors

    # Convert base data once for worker transport.
    # Avoid repeating base_data.tolist() per chunk (huge memory overhead).
    if use_numpy:
        base_list = base_data.tolist()
    else:
        base_list = base_data

    os.makedirs(data_dir, exist_ok=True)

    # Build chunk task list, skipping already-completed chunks (resume support)
    all_chunk_paths = []
    pending_tasks = []
    skipped = 0

    for chunk_id in range(num_chunks):
        q_start = chunk_id * chunk_size
        q_end = min(q_start + chunk_size, num_queries)
        cp = _chunk_path(data_dir, chunk_id)
        all_chunk_paths.append(cp)

        if os.path.isfile(cp):
            # Validate existing chunk can be reused for current params.
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if _is_valid_chunk(existing, q_start, q_end, top_k):
                    skipped += 1
                    continue
            except (json.JSONDecodeError, OSError):
                pass  # Corrupted — recompute

        if use_numpy:
            query_chunk = query_data[q_start:q_end].tolist()
        else:
            query_chunk = query_data[q_start:q_end]

        pending_tasks.append(
            (base_list, query_chunk, top_k, chunk_id, q_start, cp, use_numpy)
        )

    if skipped > 0:
        print(f"\nResuming: {skipped}/{num_chunks} chunks already completed, "
              f"{len(pending_tasks)} remaining")

    print(f"\nComputing ground truth (top-{top_k}) for {num_queries} queries "
          f"against {len(base_data)} base vectors "
          f"in {num_chunks} chunks (chunk_size={chunk_size})...")

    start = time.time()

    if pending_tasks:
        if workers <= 1:
            # Sequential processing
            for task in pending_tasks:
                _process_chunk_to_disk(task)
        else:
            # Parallel processing
            effective_workers = min(workers, len(pending_tasks))
            print(f"  Dispatching {effective_workers} worker processes "
                  f"for {len(pending_tasks)} chunks...")
            with Pool(processes=effective_workers) as pool:
                pool.map(_process_chunk_to_disk, pending_tasks)

    elapsed = time.time() - start
    print(f"\nBrute-force search completed in {elapsed:.1f}s")

    # Merge all chunk files into final output
    print(f"Merging {num_chunks} chunks into {output_path}...")
    _merge_chunk_files(all_chunk_paths, output_path, num_queries)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Written {output_path} ({size_mb:.1f} MB, {num_queries} entries)")

    # Clean up chunk files
    _cleanup_chunk_files(all_chunk_paths)
    print("Chunk files cleaned up.")


if __name__ == "__main__":
    main()
