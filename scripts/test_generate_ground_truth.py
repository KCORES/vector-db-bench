#!/usr/bin/env python3
"""Unit tests for generate_ground_truth.py brute-force KNN computation."""

import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_ground_truth import (
    read_fvecs,
    compute_ground_truth,
    _l2_distance_squared_python,
    _is_valid_chunk,
)


def write_fvecs(filepath, vectors):
    """Write vectors in fvecs binary format."""
    with open(filepath, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("<i", dim))
            f.write(struct.pack(f"<{dim}f", *vec))


class TestL2DistanceSquared(unittest.TestCase):
    def test_identical_vectors(self):
        self.assertAlmostEqual(_l2_distance_squared_python([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_known_distance(self):
        # distance([0,0], [3,4]) = 5, squared = 25
        self.assertAlmostEqual(_l2_distance_squared_python([0.0, 0.0], [3.0, 4.0]), 25.0)

    def test_single_dimension(self):
        self.assertAlmostEqual(_l2_distance_squared_python([5.0], [2.0]), 9.0)


class TestComputeGroundTruth(unittest.TestCase):
    def test_basic_knn(self):
        """Query is closest to base[0], then base[1], then base[2]."""
        base = [[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]]
        query = [[0.1, 0.1]]
        result = compute_ground_truth(base, query, top_k=2)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["query_id"], 0)
        self.assertEqual(len(result[0]["neighbors"]), 2)
        self.assertEqual(result[0]["neighbors"][0], 0)  # closest
        self.assertEqual(result[0]["neighbors"][1], 1)  # second closest

    def test_top_k_equals_base_size(self):
        """When top_k == len(base), all IDs should be returned."""
        base = [[0.0], [5.0], [3.0]]
        query = [[1.0]]
        result = compute_ground_truth(base, query, top_k=3)

        self.assertEqual(len(result[0]["neighbors"]), 3)
        # Sorted by distance: base[0]=1, base[2]=4, base[1]=16
        self.assertEqual(result[0]["neighbors"], [0, 2, 1])

    def test_multiple_queries(self):
        base = [[0.0, 0.0], [10.0, 10.0]]
        queries = [[0.0, 0.1], [9.9, 10.0]]
        result = compute_ground_truth(base, queries, top_k=1)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["neighbors"], [0])  # first query closer to base[0]
        self.assertEqual(result[1]["neighbors"], [1])  # second query closer to base[1]

    def test_output_format(self):
        base = [[1.0, 2.0], [3.0, 4.0]]
        query = [[1.5, 2.5]]
        result = compute_ground_truth(base, query, top_k=2)

        self.assertIn("query_id", result[0])
        self.assertIn("neighbors", result[0])
        self.assertIsInstance(result[0]["query_id"], int)
        self.assertIsInstance(result[0]["neighbors"], list)
        for nid in result[0]["neighbors"]:
            self.assertIsInstance(nid, int)

    def test_top_k_one(self):
        """Top-1 should return the single nearest neighbor."""
        base = [[0.0], [100.0], [50.0]]
        query = [[49.0]]
        result = compute_ground_truth(base, query, top_k=1)
        self.assertEqual(result[0]["neighbors"], [2])  # base[2]=50 is closest to 49

    def test_correctness_all_neighbors_closer_than_non_neighbors(self):
        """Every returned neighbor should be closer than any non-returned vector."""
        base = [[float(i)] for i in range(20)]
        query = [[7.5]]
        top_k = 5
        result = compute_ground_truth(base, query, top_k=top_k)

        neighbors = set(result[0]["neighbors"])
        q = query[0]

        max_neighbor_dist = max(
            _l2_distance_squared_python(q, base[n]) for n in neighbors
        )
        for i in range(len(base)):
            if i not in neighbors:
                d = _l2_distance_squared_python(q, base[i])
                self.assertGreaterEqual(
                    d, max_neighbor_dist,
                    f"Non-neighbor {i} (dist={d}) is closer than a returned neighbor "
                    f"(max_dist={max_neighbor_dist})"
                )


class TestReadFvecsInScript(unittest.TestCase):
    """Verify the inline read_fvecs works correctly."""

    def test_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
            path = f.name
            write_fvecs(path, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        try:
            result = read_fvecs(path)
            self.assertEqual(len(result), 2)
            self.assertAlmostEqual(result[0][0], 1.0)
            self.assertAlmostEqual(result[1][2], 6.0)
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_fvecs("/nonexistent/path.fvecs")


class TestChunkValidation(unittest.TestCase):
    def test_chunk_validation_rejects_wrong_top_k(self):
        chunk = [
            {"query_id": 10, "neighbors": [1, 2]},
            {"query_id": 11, "neighbors": [3, 4]},
        ]
        self.assertFalse(_is_valid_chunk(chunk, 10, 12, top_k=1))

    def test_chunk_validation_rejects_wrong_query_id(self):
        chunk = [
            {"query_id": 10, "neighbors": [1]},
            {"query_id": 99, "neighbors": [2]},
        ]
        self.assertFalse(_is_valid_chunk(chunk, 10, 12, top_k=1))

    def test_chunk_validation_accepts_matching_chunk(self):
        chunk = [
            {"query_id": 10, "neighbors": [1]},
            {"query_id": 11, "neighbors": [2]},
        ]
        self.assertTrue(_is_valid_chunk(chunk, 10, 12, top_k=1))


class TestEndToEndSmall(unittest.TestCase):
    """End-to-end test: write fvecs, compute ground truth, verify JSON output."""

    def test_small_dataset(self):
        tmpdir = tempfile.mkdtemp()
        try:
            # Create small base and query fvecs
            base = [[float(i), float(i + 1)] for i in range(10)]
            queries = [[0.5, 1.5], [8.5, 9.5]]

            base_path = os.path.join(tmpdir, "sift_base.fvecs")
            query_path = os.path.join(tmpdir, "sift_query.fvecs")
            write_fvecs(base_path, base)
            write_fvecs(query_path, queries)

            # Compute ground truth
            base_loaded = read_fvecs(base_path)
            query_loaded = read_fvecs(query_path)
            result = compute_ground_truth(base_loaded, query_loaded, top_k=3)

            # Verify structure
            self.assertEqual(len(result), 2)
            for entry in result:
                self.assertIn("query_id", entry)
                self.assertIn("neighbors", entry)
                self.assertEqual(len(entry["neighbors"]), 3)

            # Query [0.5, 1.5] should be closest to base[0]=[0,1] and base[1]=[1,2]
            self.assertEqual(result[0]["neighbors"][0], 0)
            self.assertEqual(result[0]["neighbors"][1], 1)

            # Query [8.5, 9.5] should be closest to base[8]=[8,9] and base[9]=[9,10]
            self.assertIn(8, result[1]["neighbors"][:2])
            self.assertIn(9, result[1]["neighbors"][:2])

            # Verify JSON serializable
            json_str = json.dumps(result)
            parsed = json.loads(json_str)
            self.assertEqual(parsed, result)
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_resume_recompute_when_top_k_changes(self):
        tmpdir = tempfile.mkdtemp()
        try:
            base = [[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]]
            queries = [[0.1, 0.1], [9.9, 9.9]]

            write_fvecs(os.path.join(tmpdir, "sift_base.fvecs"), base)
            write_fvecs(os.path.join(tmpdir, "sift_query.fvecs"), queries)

            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_ground_truth.py")

            subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--data-dir", tmpdir,
                    "--top-k", "2",
                    "--chunk-size", "1",
                    "--workers", "1",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            with open(os.path.join(tmpdir, "ground_truth.json"), "r", encoding="utf-8") as f:
                stale = json.load(f)

            for chunk_id, row in enumerate(stale):
                with open(
                    os.path.join(tmpdir, f"ground_truth_chunk_{chunk_id}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump([row], f)

            subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--data-dir", tmpdir,
                    "--top-k", "1",
                    "--chunk-size", "1",
                    "--workers", "1",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            with open(os.path.join(tmpdir, "ground_truth.json"), "r", encoding="utf-8") as f:
                updated = json.load(f)

            self.assertTrue(all(len(entry["neighbors"]) == 1 for entry in updated))
        finally:
            import shutil
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
