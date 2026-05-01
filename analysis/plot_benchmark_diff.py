#!/usr/bin/env python3
"""
plot_benchmark_diff.py — 20B vs 120B fine-tuning effect (FT − base) per task.

Two horizontal-bar panels, one per model size, sharing task order. Bars colored
green for gain (FT > base) and red for loss (FT < base). The aggregate rows
("bbh", "gpqa") are emphasized.

Usage:
    python analysis/plot_benchmark_diff.py \
        --comparison-json overnight_results/latest/benchmark_comparison.json \
        --output overnight_results/latest/benchmark_diff_20b_vs_120b.png
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from analysis.plots import COLORS, setup_style


AGGREGATE_TASKS = {"bbh", "gpqa"}


def _domain(task):
    if task in {"college_physics", "high_school_physics", "astronomy"}:
        return "MMLU"
    if task.startswith("gpqa"):
        return "GPQA"
    if task.startswith("bbh"):
        return "BBH"
    return "other"


def plot_diff(comparison_path, output_path):
    with open(comparison_path) as f:
        data = json.load(f)["tasks"]

    # Compute deltas and sort tasks by 120B delta (descending: gains on top).
    rows = []
    for task, scores in data.items():
        d20 = scores["ft_20b"] - scores["base_20b"]
        d120 = scores["ft_120b"] - scores["base_120b"]
        rows.append((task, d20, d120, _domain(task)))
    rows.sort(key=lambda r: r[2], reverse=True)

    tasks = [r[0] for r in rows]
    d20 = np.array([r[1] for r in rows])
    d120 = np.array([r[2] for r in rows])

    setup_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 0.30 * len(tasks) + 1.5), sharey=True
    )

    y = np.arange(len(tasks))
    gain = COLORS["accent"]   # green
    loss = COLORS["base"]     # red

    for ax, deltas, title in [(ax1, d20, "gpt-oss-20B"),
                              (ax2, d120, "gpt-oss-120B")]:
        colors = [gain if v >= 0 else loss for v in deltas]
        ax.barh(y, deltas, color=colors, alpha=0.85, edgecolor="none")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(r"$\Delta$ accuracy [pp]  (FT $-$ base)")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.3, lw=0.6)

        for yi, (v, task) in enumerate(zip(deltas, tasks)):
            offset = 0.4 if v >= 0 else -0.4
            ha = "left" if v >= 0 else "right"
            ax.text(v + offset, yi, f"{v:+.1f}",
                    va="center", ha=ha, fontsize=8,
                    color="black", alpha=0.75)

        net = deltas.sum() / len(deltas)
        ax.text(0.97, 0.02, f"mean Δ = {net:+.2f} pp",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="gray", alpha=0.85))

    ax1.set_yticks(y)
    labels = []
    for task in tasks:
        label = task.replace("bbh_", "").replace("leaderboard_", "")
        if task in AGGREGATE_TASKS:
            label = f"⟨{label}⟩"
        labels.append(label)
    ax1.set_yticklabels(labels, fontsize=8)

    for tick, task in zip(ax1.get_yticklabels(), tasks):
        if task in AGGREGATE_TASKS:
            tick.set_fontweight("bold")

    xmax = max(abs(d20).max(), abs(d120).max()) + 6
    ax1.set_xlim(-xmax, xmax)
    ax2.set_xlim(-xmax, xmax)
    ax1.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved {output_path}")
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--comparison-json", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    plot_diff(args.comparison_json, args.output)
