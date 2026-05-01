#!/usr/bin/env python3
"""
plot_pareto_money.py — Pareto-aware money plot for the multi-objective
autoresearch loop (see experiments/pareto_loop.py).

Two panels:
  • Top — scatter of (MAE, MCQ score) across all trials, colored by trial
    order, Pareto front connected with a line. Failed trials shown as ✕.
  • Bottom — running-best convergence: lowest MAE seen so far AND highest
    MCQ score so far, plotted vs trial number on twin axes.

Usage:
    python analysis/plot_pareto_money.py \
        --results experiments/autoresearch_runs/pareto/results.jsonl \
        --output  plots/autoresearch_pareto.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.plots import COLORS, setup_style


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_pareto(jsonl_path, output_path):
    records = load_jsonl(jsonl_path)
    if not records:
        print(f"[plot_pareto] no records in {jsonl_path}")
        return

    valid = [r for r in records if r.get("mae") is not None and r.get("mcq") is not None]
    failed = [r for r in records if r not in valid]

    setup_style()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 2]}
    )

    # ── Panel 1: scatter + Pareto front ────────────────────────────────
    if valid:
        trials = np.array([r["trial"] for r in valid])
        mae = np.array([r["mae"] for r in valid])
        mcq = np.array([r["mcq"] for r in valid])
        is_pareto = np.array([bool(r.get("pareto", False)) for r in valid])

        sc = ax1.scatter(mae, mcq, c=trials, cmap="viridis", s=55,
                         alpha=0.85, edgecolors="white", linewidths=0.6,
                         zorder=4, label="Trial")
        cb = fig.colorbar(sc, ax=ax1, pad=0.01, fraction=0.04)
        cb.set_label("Trial #", fontsize=9)

        # Pareto front: sort by MAE ascending, plot connecting line
        if is_pareto.any():
            front_mae = mae[is_pareto]
            front_mcq = mcq[is_pareto]
            front_trials = trials[is_pareto]
            order = np.argsort(front_mae)
            ax1.plot(front_mae[order], front_mcq[order], "-",
                     color=COLORS["base"], lw=1.5, alpha=0.7, zorder=3,
                     label=f"Pareto front (n={is_pareto.sum()})")
            ax1.scatter(front_mae, front_mcq, marker="D", s=110,
                        facecolors="none", edgecolors=COLORS["base"],
                        linewidths=1.6, zorder=5)
            for tm, tcq, ti in zip(front_mae, front_mcq, front_trials):
                ax1.annotate(f"#{int(ti)}", (tm, tcq),
                             textcoords="offset points", xytext=(7, 6),
                             fontsize=8, color=COLORS["base"], alpha=0.9)

    if failed:
        # Place crashes off-axis at MAE=ax1 right edge, MCQ=bottom
        x = ax1.get_xlim()[1] if valid else 0.2
        for r in failed:
            ax1.plot(x, ax1.get_ylim()[0] if valid else 0,
                     "x", color="#ff6666", ms=9, alpha=0.6, zorder=2)

    ax1.set_xlabel("Redshift MAE  (lower is better)")
    ax1.set_ylabel("MCQ score [%]  (sci_reasoning + general_qa) / 2")
    ax1.set_title(f"Pareto trace — {len(records)} trials "
                  f"({len(valid)} valid, {len(failed)} failed)")
    ax1.grid(True, alpha=0.3, lw=0.6)
    ax1.legend(loc="best", fontsize=9)

    # ── Panel 2: convergence ───────────────────────────────────────────
    if valid:
        order = np.argsort(trials)
        t_sorted = trials[order]
        mae_sorted = mae[order]
        mcq_sorted = mcq[order]

        running_best_mae = np.minimum.accumulate(mae_sorted)
        running_best_mcq = np.maximum.accumulate(mcq_sorted)

        color_mae = COLORS["primary"]
        color_mcq = COLORS["accent"]

        ax2.plot(t_sorted, running_best_mae, "-o", color=color_mae,
                 lw=1.6, ms=4, label="Best MAE so far")
        ax2.set_xlabel("Trial #")
        ax2.set_ylabel("Best MAE so far", color=color_mae)
        ax2.tick_params(axis="y", labelcolor=color_mae)
        ax2.grid(True, alpha=0.3, lw=0.6)

        ax3 = ax2.twinx()
        ax3.plot(t_sorted, running_best_mcq, "-s", color=color_mcq,
                 lw=1.6, ms=4, label="Best MCQ so far")
        ax3.set_ylabel("Best MCQ score so far [%]", color=color_mcq)
        ax3.tick_params(axis="y", labelcolor=color_mcq)

        ax2.set_title("Running-best convergence (each objective independently)")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"[plot_pareto] saved {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--output",  required=True)
    args = p.parse_args()
    plot_pareto(Path(args.results), args.output)


if __name__ == "__main__":
    main()
