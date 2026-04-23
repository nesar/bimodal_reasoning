#!/usr/bin/env python3
"""
plot_fig1.py — Reproduce the Figure 1 layout from the paper:

    Top row:    Redshift scatter plots (true vs predicted) with KDE contours
    Bottom row: Benchmark retention bar chart (percent change from base model)

Usage:
    python analysis/plot_fig1.py \
        --eval_dirs "label1=path1" "label2=path2" \
        --benchmark_json path/to/benchmark_comparison.json \
        --output plots/fig1.png

    # Minimal (just redshift, no benchmarks yet):
    python analysis/plot_fig1.py \
        --eval_dirs "20B compact=overnight_results/latest/phase2_eval_20b_compact" \
        --output plots/fig1.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.plots import setup_style, COLORS, _density_contour


def load_predictions(eval_dir):
    pred_file = Path(eval_dir) / "raw_predictions.jsonl"
    if not pred_file.exists():
        return np.array([]), np.array([])
    records = [json.loads(line) for line in open(pred_file)]
    z_true = np.array([r["z_true"] if r["z_true"] is not None else np.nan for r in records])
    z_pred = np.array([r["z_pred"] if r["z_pred"] is not None else np.nan for r in records])
    return z_true, z_pred


def panel_redshift(ax, z_true, z_pred, label, color):
    mask = np.isfinite(z_true) & np.isfinite(z_pred)
    zt, zp = z_true[mask], z_pred[mask]
    n_valid = len(zt)
    n_total = len(z_true)

    if n_valid == 0:
        ax.text(0.5, 0.5, f"No valid\npredictions\n(0/{n_total})",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title(label, fontsize=11, fontweight="bold")
        return

    mae = float(np.mean(np.abs(zt - zp)))

    if n_valid >= 20:
        _density_contour(ax, zt, zp, color)
    ax.scatter(zt, zp, s=12, color=color, alpha=0.5, edgecolors="none", zorder=5)

    lo = min(zt.min(), zp.min()) - 0.02
    hi = max(zt.max(), zp.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"True Redshift", fontsize=10)
    ax.set_ylabel(r"Predicted Redshift", fontsize=10)

    info = f"MAE = {mae:.4f}\n$n$ = {n_valid}/{n_total}"
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, lw=0.6)


def panel_benchmarks(ax, benchmark_json_path):
    if not benchmark_json_path or not Path(benchmark_json_path).exists():
        ax.text(0.5, 0.5, "Benchmark eval\nnot yet available\n(run_benchmarks.sh pending)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11,
                color="gray")
        ax.set_title("Performance Transition After Fine-tuning", fontsize=11, fontweight="bold")
        ax.set_xlabel("Percent Change (%)", fontsize=10)
        return

    with open(benchmark_json_path) as f:
        data = json.load(f)

    tasks_dict = data.get("tasks", {})
    task_names = sorted(tasks_dict.keys())
    base_scores = [tasks_dict[t]["base"] for t in task_names]
    ft_scores = [tasks_dict[t]["finetuned"] for t in task_names]

    # Compute percent change
    pct_change = [(f - b) for b, f in zip(base_scores, ft_scores)]

    # Clean up task names for display
    display_names = []
    for t in task_names:
        t = t.replace("college_physics", "MMLU College Physics")
        t = t.replace("high_school_physics", "MMLU HS Physics")
        t = t.replace("astronomy", "MMLU Astronomy")
        t = t.replace("gpqa", "GPQA")
        t = t.replace("bbh", "Big-Bench Hard")
        display_names.append(t)

    y_pos = np.arange(len(display_names))
    colors = [COLORS["accent"] if d >= 0 else COLORS["base"] for d in pct_change]

    ax.barh(y_pos, pct_change, color=colors, alpha=0.8, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(-5, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.axvline(-15, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Percent Change (%)", fontsize=10)
    ax.set_title("Performance Transition After Fine-tuning", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, lw=0.6)
    ax.invert_yaxis()

    # Annotate values
    for i, (d, name) in enumerate(zip(pct_change, display_names)):
        ha = "left" if d >= 0 else "right"
        offset = 0.3 if d >= 0 else -0.3
        ax.text(d + offset, i, f"{d:+.1f}%", va="center", ha=ha, fontsize=8)


def make_fig1(eval_dirs, benchmark_json, output_path):
    setup_style()

    n_scatter = max(len(eval_dirs), 1)
    has_benchmarks = benchmark_json and Path(benchmark_json).exists()

    fig = plt.figure(figsize=(5.5 * n_scatter, 10))
    gs = GridSpec(2, n_scatter, figure=fig, height_ratios=[1, 0.7],
                  hspace=0.35, wspace=0.3)

    # Top row: redshift scatter panels
    scatter_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
    for i, (label, eval_dir) in enumerate(eval_dirs.items()):
        ax = fig.add_subplot(gs[0, i])
        z_true, z_pred = load_predictions(eval_dir)
        panel_redshift(ax, z_true, z_pred, label, scatter_colors[i % len(scatter_colors)])

    # Bottom row: benchmark bar chart (spans full width)
    ax_bench = fig.add_subplot(gs[1, :])
    panel_benchmarks(ax_bench, benchmark_json)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Figure 1 saved to {output_path}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dirs", nargs="+", required=True,
                        help="label=path pairs for eval result dirs")
    parser.add_argument("--benchmark_json", default=None,
                        help="Path to benchmark_comparison.json")
    parser.add_argument("--output", default="plots/fig1.png")
    args = parser.parse_args()

    eval_dirs = {}
    for e in args.eval_dirs:
        label, path = e.split("=", 1)
        eval_dirs[label] = path

    make_fig1(eval_dirs, args.benchmark_json, args.output)


if __name__ == "__main__":
    main()
