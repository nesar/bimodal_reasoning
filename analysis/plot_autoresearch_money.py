#!/usr/bin/env python3
"""
plot_autoresearch_money.py — Karpathy-style money plot for the fine-tuning autoresearch loop.

Shows: kept experiments (green), discarded (gray), running best line,
and annotated descriptions of what changed at each step.

Usage:
    python analysis/plot_autoresearch_money.py \
        --results experiments/autoresearch_runs/results.tsv \
        --output plots/autoresearch_money.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.plots import setup_style, COLORS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="experiments/autoresearch_runs/results.tsv")
    parser.add_argument("--output", default="plots/autoresearch_money.png")
    args = parser.parse_args()

    # Parse results.tsv
    experiments = []
    with open(args.results) as f:
        header = next(f).strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                experiments.append({
                    "experiment": int(parts[0]),
                    "mae": float(parts[1]),
                    "train_loss": float(parts[2]),
                    "status": parts[3],
                    "description": parts[4],
                })

    if not experiments:
        print("No experiments found in results.tsv")
        return

    setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    n_kept = sum(1 for e in experiments if e["status"] == "keep")
    n_total = len(experiments)

    # Plot discarded (gray, background)
    for e in experiments:
        if e["status"] == "discard" and e["mae"] < 10:
            ax.plot(e["experiment"], e["mae"], "o", color="#cccccc", ms=7,
                    zorder=2, alpha=0.7)

    # Plot crashed
    for e in experiments:
        if e["status"] == "crash":
            ax.plot(e["experiment"], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
                    "x", color="#ff6666", ms=8, zorder=2, alpha=0.5)

    # Plot kept (green)
    kept = [e for e in experiments if e["status"] == "keep"]
    if kept:
        kept_x = [e["experiment"] for e in kept]
        kept_y = [e["mae"] for e in kept]
        ax.plot(kept_x, kept_y, "o", color="#2ca02c", ms=9, zorder=5, label="Kept")

        # Running best line (step function)
        running_best_x = [kept_x[0]]
        running_best_y = [kept_y[0]]
        current_best = kept_y[0]
        for x, y in zip(kept_x[1:], kept_y[1:]):
            # Horizontal line to this x
            running_best_x.append(x)
            running_best_y.append(current_best)
            # Drop to new y
            current_best = min(current_best, y)
            running_best_x.append(x)
            running_best_y.append(current_best)

        ax.plot(running_best_x, running_best_y, "-", color="#2ca02c", lw=2,
                alpha=0.7, zorder=4, label="Running best")

    # Annotate kept experiments with descriptions
    for e in kept:
        desc = e["description"]
        if len(desc) > 40:
            desc = desc[:37] + "..."
        ax.annotate(desc, (e["experiment"], e["mae"]),
                     textcoords="offset points", xytext=(8, -12),
                     fontsize=7, color="#2ca02c", alpha=0.85,
                     rotation=15, ha="left",
                     arrowprops=dict(arrowstyle="-", color="#2ca02c",
                                      alpha=0.3, lw=0.5))

    # Annotate discarded experiments (smaller, gray)
    for e in experiments:
        if e["status"] == "discard" and e["mae"] < 10:
            desc = e["description"]
            if len(desc) > 30:
                desc = desc[:27] + "..."
            ax.annotate(desc, (e["experiment"], e["mae"]),
                         textcoords="offset points", xytext=(5, 5),
                         fontsize=5.5, color="#999999", alpha=0.6,
                         rotation=10, ha="left")

    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_ylabel("Redshift MAE (lower is better)", fontsize=11)
    ax.set_title(f"Autoresearch Progress: {n_total} Experiments, "
                 f"{n_kept} Kept Improvements", fontsize=12, fontweight="bold")

    # Add discarded to legend
    ax.plot([], [], "o", color="#cccccc", ms=7, label="Discarded")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, lw=0.6)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"Money plot saved to {args.output}")


if __name__ == "__main__":
    main()
