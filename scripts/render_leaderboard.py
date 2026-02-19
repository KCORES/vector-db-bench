#!/usr/bin/env python3
"""
Render leaderboard.json to a bar chart PNG image.

Usage:
    python3 scripts/render_leaderboard.py
    python3 scripts/render_leaderboard.py --input results/leaderboard.json --output assets/images/leaderboard.png
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_and_deduplicate(path: str, show_all: bool = False) -> list[dict]:
    """Load leaderboard JSON.

    If show_all is False (default), keep only the highest-QPS entry per model.
    If show_all is True, keep every entry with recall_passed=true.
    """
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if show_all:
        passed = [e for e in entries if e.get("recall_passed", False)]
        return sorted(passed, key=lambda x: x["qps"], reverse=True)

    best: dict[str, dict] = {}
    for e in entries:
        if not e.get("recall_passed", False):
            continue
        name = e["model_name"]
        if name not in best or e["qps"] > best[name]["qps"]:
            best[name] = e

    # Sort by QPS descending
    return sorted(best.values(), key=lambda x: x["qps"], reverse=True)


def render_chart(entries: list[dict], output_path: str, watermark_path: str | None = None, show_all: bool = False):
    """Render a horizontal bar chart and save to output_path."""
    if show_all:
        names = [f"{e['model_name']}\n({e.get('entry_dir', '')})" for e in entries]
    else:
        names = [e["model_name"] for e in entries]
    qps_values = [e["qps"] for e in entries]
    recalls = [e.get("recall", 0) for e in entries]

    # Colors per bar
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(names), 3)))

    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 1.2)))

    bars = ax.barh(range(len(names)), qps_values, color=colors[: len(names)], height=0.6, edgecolor="white")

    # Labels on bars
    for i, (bar, qps, recall) in enumerate(zip(bars, qps_values, recalls)):
        ax.text(
            bar.get_width() + max(qps_values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{qps:.1f} QPS  (recall {recall:.2%})",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Queries Per Second (QPS)", fontsize=12)
    ax.set_title("Vector DB Bench Leaderboard", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, max(qps_values) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.2, right=0.95, top=0.90, bottom=0.12)

    # Render chart to buffer, then use Pillow for pixel-precise padding & watermark
    import io
    dpi = 150
    padding_px = 50

    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    chart_img = Image.open(buf).convert("RGBA")
    cw, ch = chart_img.size

    # New canvas with 50px top + 50px bottom white padding
    canvas = Image.new("RGBA", (cw, ch + padding_px * 2), (255, 255, 255, 255))
    canvas.paste(chart_img, (0, padding_px))

    # Watermark: scale to 1/2, position 10px from bottom-right
    if watermark_path and os.path.isfile(watermark_path):
        logo = Image.open(watermark_path).convert("RGBA")
        logo = logo.resize((logo.width // 2, logo.height // 2), Image.LANCZOS)
        margin = 10
        canvas.paste(logo, (canvas.width - logo.width - margin, canvas.height - logo.height - margin), logo)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path, dpi=(dpi, dpi))
    print(f"Saved chart to {output_path}")


def main():
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Render leaderboard bar chart")
    parser.add_argument("--input", default=str(repo_root / "results" / "leaderboard.json"))
    parser.add_argument("--output", default=None, help="Output image path (default depends on --all flag)")
    parser.add_argument("--watermark", default=str(repo_root / "assets" / "images" / "kcores-llm-arena-logo-black.png"))
    parser.add_argument("--all", action="store_true", help="Show all entries instead of only the best per model")
    args = parser.parse_args()

    if args.output is None:
        if args.all:
            args.output = str(repo_root / "assets" / "images" / "leaderboard_all.png")
        else:
            args.output = str(repo_root / "assets" / "images" / "leaderboard.png")

    entries = load_and_deduplicate(args.input, show_all=args.all)
    if not entries:
        print("No valid entries found in leaderboard.")
        return

    render_chart(entries, args.output, args.watermark, show_all=args.all)


if __name__ == "__main__":
    main()
