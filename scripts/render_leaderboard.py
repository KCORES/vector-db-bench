#!/usr/bin/env python3
"""
Render leaderboard.json to a bar chart + step line chart PNG image.

Usage:
    python3 scripts/render_leaderboard.py
    python3 scripts/render_leaderboard.py --input results/leaderboard.json --output assets/images/leaderboard.png
"""

import argparse
import glob
import io
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_benchmark_progression(entry_dir: str, leaderboard_dir: str) -> list[float]:
    """Load QPS values from benchmarks/benchmark_xxx.json files in order."""
    bench_dir = os.path.join(leaderboard_dir, entry_dir, "benchmarks")
    if not os.path.isdir(bench_dir):
        return []
    qps_list = []
    for f in sorted(glob.glob(os.path.join(bench_dir, "benchmark_*.json"))):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                b = json.load(fh)
            qps_list.append(b.get("qps", 0))
        except (json.JSONDecodeError, IOError):
            qps_list.append(0)
    return qps_list


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




def render_chart(entries: list[dict], output_path: str, watermark_path: str | None = None,
                 show_all: bool = False, leaderboard_dir: str = "./leaderboard"):
    """Render a horizontal bar chart (left) + per-model step line charts (right)."""
    n = len(entries)
    if show_all:
        names = [f"{e['model_name']}\n({e.get('entry_dir', '')})" for e in entries]
    else:
        names = [e["model_name"] for e in entries]
    qps_values = [e["qps"] for e in entries]
    recalls = [e.get("recall", 0) for e in entries]

    # Load benchmark progression for each entry
    progressions = []
    for e in entries:
        prog = load_benchmark_progression(e.get("entry_dir", ""), leaderboard_dir)
        progressions.append(prog)

    # Build a stable color map: same model_name -> same color
    unique_models = list(dict.fromkeys(e["model_name"] for e in entries))
    palette = plt.cm.Set2(np.linspace(0, 1, max(len(unique_models), 3)))
    model_color = {name: palette[i] for i, name in enumerate(unique_models)}
    colors = [model_color[e["model_name"]] for e in entries]

    row_height = 1.2
    fig_h = max(4, n * row_height + 1.5)
    fig = plt.figure(figsize=(20, fig_h))

    # GridSpec: leave right=0.78 so right-side labels have room inside the figure
    gs = fig.add_gridspec(n, 2, width_ratios=[1, 1.2], wspace=0.35,
                          left=0.15, right=0.78, top=0.90, bottom=0.08)

    # === Left: Bar chart spanning all rows ===
    ax_bar = fig.add_subplot(gs[:, 0])
    bars = ax_bar.barh(range(n), qps_values, color=colors, height=0.6, edgecolor="white")

    for bar, qps, recall in zip(bars, qps_values, recalls):
        ax_bar.text(
            bar.get_width() + max(qps_values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{qps:.1f} QPS  (recall {recall:.2%})",
            va="center", fontsize=10, fontweight="bold",
        )

    ax_bar.set_yticks(range(n))
    ax_bar.set_yticklabels(names, fontsize=11)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Queries Per Second (QPS)", fontsize=12)
    ax_bar.set_title("Leaderboard", fontsize=14, fontweight="bold", pad=15)
    ax_bar.set_xlim(0, max(qps_values) * 1.35)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # === Right: One step line chart per model ===
    # Add a shared subtitle for the right column
    ax_top = None
    for i in range(n):
        ax = fig.add_subplot(gs[i, 1])
        if i == 0:
            ax_top = ax
            ax.set_title("QPS Progression per Iteration", fontsize=14, fontweight="bold", pad=15)
        prog = progressions[i]
        c = colors[i]

        if prog:
            steps = list(range(1, len(prog) + 1))
            ax.step(steps, prog, where="mid", color=c, linewidth=2)
            ax.plot(steps, prog, "o", color=c, markersize=4)
            ax.set_xlim(0.5, len(prog) + 0.5)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

        # Only bottom chart gets x-label
        if i == n - 1:
            ax.set_xlabel("Iteration", fontsize=9)
        else:
            ax.set_xticklabels([])

        # Label to the right of the mini chart, using annotate to avoid clipping
        short_name = entries[i]["model_name"]
        if show_all:
            short_name += f"\n({entries[i].get('entry_dir', '')})"
        ax.annotate(short_name, xy=(1.02, 0.5), xycoords="axes fraction",
                    fontsize=8, va="center", ha="left", color=c, fontweight="bold")

    fig.suptitle("Vector DB Bench Leaderboard", fontsize=16, fontweight="bold", y=0.98)

    # Render to buffer then add padding & watermark via Pillow
    dpi = 150
    padding_top = 50
    padding_bottom = 50
    padding_left = 100
    padding_right = 50

    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    chart_img = Image.open(buf).convert("RGBA")
    cw, ch = chart_img.size

    canvas = Image.new("RGBA",
                       (cw + padding_left + padding_right, ch + padding_top + padding_bottom),
                       (255, 255, 255, 255))
    canvas.paste(chart_img, (padding_left, padding_top))

    if watermark_path and os.path.isfile(watermark_path):
        logo = Image.open(watermark_path).convert("RGBA")
        logo = logo.resize((logo.width // 2, logo.height // 2), Image.LANCZOS)
        margin = 10
        canvas.paste(logo, (canvas.width - logo.width - margin,
                            canvas.height - logo.height - margin), logo)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path, dpi=(dpi, dpi))
    print(f"Saved chart to {output_path}")




def main():
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Render leaderboard bar chart")
    parser.add_argument("--input", default=str(repo_root / "results" / "leaderboard.json"))
    parser.add_argument("--output", default=None, help="Output image path (default depends on --all flag)")
    parser.add_argument("--watermark", default=str(repo_root / "assets" / "images" / "kcores-llm-arena-logo-black.png"))
    parser.add_argument("--leaderboard-dir", default=str(repo_root / "leaderboard"), help="Path to leaderboard directory")
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

    render_chart(entries, args.output, args.watermark, show_all=args.all,
                 leaderboard_dir=args.leaderboard_dir)


if __name__ == "__main__":
    main()
