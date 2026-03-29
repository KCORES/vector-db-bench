#!/usr/bin/env python3
"""
Rebuild the leaderboard from all entries in the leaderboard/ directory.

Scans each subdirectory under leaderboard/ and extracts the best benchmark
result using the following priority:
  1. best_benchmark from eval_log.json (agent-tracked best QPS with passing recall)
  2. last_benchmark from eval_log.json (final finish() result)
  3. benchmark from final_report.json (if exists)
  4. Scan benchmarks/*.json for the highest QPS with passing recall

Model name is read from the first line of agent_log.jsonl (session_start event).

Usage:
    python scripts/rebuild_leaderboard.py [--leaderboard-dir ./leaderboard] [--output ./results/leaderboard.json]
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone


def read_model_name(entry_dir: str) -> str | None:
    """Read model name from the first line of agent_log.jsonl."""
    log_path = os.path.join(entry_dir, "agent_log.jsonl")
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                return None
            data = json.loads(first_line)
            if data.get("event") == "session_start":
                return data.get("model")
    except (json.JSONDecodeError, IOError):
        pass
    return None


def scan_benchmark_files(entry_dir: str) -> dict | None:
    """Scan benchmarks/*.json for the highest QPS with passing recall."""
    bench_dir = os.path.join(entry_dir, "benchmarks")
    if not os.path.isdir(bench_dir):
        return None
    best = None
    for f in sorted(glob.glob(os.path.join(bench_dir, "benchmark_*.json"))):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                b = json.load(fh)
            if b.get("recall_passed", False) and b.get("qps", 0) > 0:
                if best is None or b["qps"] > best["qps"]:
                    best = b
        except (json.JSONDecodeError, KeyError, IOError):
            continue
    return best


def extract_best_result(entry_dir: str) -> tuple[dict | None, str]:
    """
    Extract the best benchmark result from an entry directory.
    Returns (result_dict, source_name).
    """
    chosen = None
    source = "none"

    # Priority 1: best_benchmark from eval_log.json
    eval_log_path = os.path.join(entry_dir, "eval_log.json")
    if os.path.isfile(eval_log_path):
        try:
            with open(eval_log_path, "r", encoding="utf-8") as f:
                eval_log = json.load(f)

            best_bench = eval_log.get("best_benchmark")
            if best_bench and best_bench.get("recall_passed", False) and best_bench.get("qps", 0) > 0:
                chosen = best_bench
                source = "best_benchmark"

            # Priority 2: last_benchmark
            last_bench = eval_log.get("last_benchmark")
            if last_bench and last_bench.get("recall_passed", False) and last_bench.get("qps", 0) > 0:
                if chosen is None or last_bench["qps"] > chosen["qps"]:
                    chosen = last_bench
                    source = "last_benchmark"
        except (json.JSONDecodeError, IOError):
            pass

    # Priority 3: final_report.json
    report_path = os.path.join(entry_dir, "final_report.json")
    if os.path.isfile(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            bench = report.get("benchmark")
            if bench and bench.get("recall_passed", False) and bench.get("qps", 0) > 0:
                if chosen is None or bench["qps"] > chosen["qps"]:
                    chosen = bench
                    source = "final_report"
        except (json.JSONDecodeError, IOError):
            pass

    # Priority 4: scan benchmark files
    scanned = scan_benchmark_files(entry_dir)
    if scanned:
        if chosen is None or scanned["qps"] > chosen["qps"]:
            chosen = scanned
            source = "benchmark_files_scan"

    return chosen, source


def count_tool_calls_from_agent_log(entry_dir: str) -> int:
    """Count tool_call events from agent_log.jsonl as a fallback."""
    log_path = os.path.join(entry_dir, "agent_log.jsonl")
    if not os.path.isfile(log_path):
        return 0
    count = 0
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("event") == "tool_call":
                        count += 1
                except json.JSONDecodeError:
                    continue
    except IOError:
        pass
    return count


def get_tool_calls_used(entry_dir: str) -> int:
    """Read tool_calls_used from eval_log.json, fallback to agent_log.jsonl."""
    eval_log_path = os.path.join(entry_dir, "eval_log.json")
    if os.path.isfile(eval_log_path):
        try:
            with open(eval_log_path, "r", encoding="utf-8") as f:
                val = json.load(f).get("tool_calls_used", 0)
                if val > 0:
                    return val
        except (json.JSONDecodeError, IOError):
            pass

    fallback = count_tool_calls_from_agent_log(entry_dir)
    if fallback > 0:
        return fallback

    return 0


def rebuild_leaderboard(leaderboard_dir: str, output_path: str):
    """Scan all entries and rebuild the leaderboard."""
    if not os.path.isdir(leaderboard_dir):
        print(f"Error: leaderboard directory not found: {leaderboard_dir}", file=sys.stderr)
        sys.exit(1)

    entries = []

    for entry_name in sorted(os.listdir(leaderboard_dir)):
        entry_dir = os.path.join(leaderboard_dir, entry_name)
        if not os.path.isdir(entry_dir):
            continue

        # Skip target/ and other non-entry dirs
        if entry_name.startswith(".") or entry_name == "target":
            continue

        model_name = read_model_name(entry_dir)
        if model_name is None:
            # Fallback: use directory name
            model_name = entry_name
            print(f"  Warning: no agent_log.jsonl in {entry_name}, using dir name as model", file=sys.stderr)

        result, source = extract_best_result(entry_dir)

        qps = 0.0
        recall = 0.0
        recall_passed = False

        if result:
            qps = result.get("qps", 0)
            recall = result.get("recall", 0)
            recall_passed = result.get("recall_passed", False)
            # Enforce recall threshold
            if recall < 0.95:
                qps = 0

        tool_calls_used = get_tool_calls_used(entry_dir)

        entry = {
            "model_name": model_name,
            "entry_dir": entry_name,
            "qps": qps,
            "recall": recall,
            "recall_passed": recall_passed,
            "tool_calls_used": tool_calls_used,
            "result_source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        entries.append(entry)

        status = "OK" if qps > 0 else "NO SCORE"
        print(f"  [{status}] {entry_name}: model={model_name}, QPS={qps:.2f}, recall={recall:.4f}, source={source}")

    # Sort: QPS descending, then recall descending
    entries.sort(key=lambda e: (-e["qps"], -e["recall"]))

    # Write leaderboard
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    print()
    print("=== Leaderboard ===")
    for i, e in enumerate(entries):
        src = f" [{e['result_source']}]" if e.get("result_source") else ""
        print(f"  {i+1}. {e['model_name']:25s}  QPS: {e['qps']:>10.2f}  Recall: {e['recall']:.4f}{src}  ({e['entry_dir']})")
    print()
    print(f"Leaderboard written to {output_path} ({len(entries)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Rebuild leaderboard from leaderboard/ entries")
    parser.add_argument(
        "--leaderboard-dir",
        default="./leaderboard",
        help="Path to leaderboard directory (default: ./leaderboard)",
    )
    parser.add_argument(
        "--output", "-o",
        default="./results/leaderboard.json",
        help="Output path for leaderboard JSON (default: ./results/leaderboard.json)",
    )
    args = parser.parse_args()

    print(f"Scanning {args.leaderboard_dir} for entries...")
    rebuild_leaderboard(args.leaderboard_dir, args.output)


if __name__ == "__main__":
    main()
