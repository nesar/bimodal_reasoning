#!/usr/bin/env python3
"""
plot_money.py — Comprehensive "money plot" for the bimodal reasoning project.

Generates a multi-panel publication figure summarizing all experiments:
  - Panel A: Training loss comparison (20B vs 120B)
  - Panel B: Redshift scatter (20B vs 120B, side by side)
  - Panel C: Benchmark comparison (base vs fine-tuned) [if data available]
  - Panel D: Summary metrics table

Usage:
    python analysis/plot_money.py --output_dir plots/summary
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.plots import setup_style, COLORS, _density_contour


def load_trainer_state(run_dir: Path) -> list[dict]:
    checkpoints = sorted(run_dir.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1]))
    for ckpt in reversed(checkpoints):
        f = ckpt / "trainer_state.json"
        if f.exists():
            with open(f) as fh:
                return json.load(fh)["log_history"]
    f = run_dir / "trainer_state.json"
    if f.exists():
        with open(f) as fh:
            return json.load(fh)["log_history"]
    return []


def load_metrics(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def panel_loss_comparison(ax, runs: dict):
    """Panel A: Training loss curves for multiple runs."""
    colors_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
    for i, (name, run_dir) in enumerate(runs.items()):
        log = load_trainer_state(Path(run_dir))
        if not log:
            continue
        steps = [e["step"] for e in log if "loss" in e]
        losses = [e["loss"] for e in log if "loss" in e]
        color = colors_cycle[i % len(colors_cycle)]
        ax.plot(steps, losses, color=color, lw=1.5, label=name, alpha=0.85)

    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.set_title("(a) Training Loss", fontsize=11, fontweight="bold")


def panel_redshift_scatter(axes, eval_dirs: dict):
    """Panel B: Side-by-side redshift scatter plots."""
    colors_cycle = [COLORS["primary"], COLORS["secondary"]]
    for i, (name, eval_dir) in enumerate(eval_dirs.items()):
        ax = axes[i] if len(axes) > 1 else axes
        pred_file = Path(eval_dir) / "raw_predictions.jsonl"
        if not pred_file.exists():
            ax.text(0.5, 0.5, f"No eval data\n({name})",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"(b{i+1}) {name}", fontsize=11, fontweight="bold")
            continue

        records = [json.loads(line) for line in open(pred_file)]
        z_true = np.array([r["z_true"] if r["z_true"] is not None else np.nan for r in records])
        z_pred = np.array([r["z_pred"] if r["z_pred"] is not None else np.nan for r in records])
        mask = np.isfinite(z_true) & np.isfinite(z_pred)
        zt, zp = z_true[mask], z_pred[mask]
        mae = float(np.mean(np.abs(zt - zp))) if len(zt) > 0 else float("nan")

        color = colors_cycle[i % len(colors_cycle)]
        if len(zt) >= 20:
            _density_contour(ax, zt, zp, color)
        ax.scatter(zt, zp, s=15, color=color, alpha=0.6, edgecolors="none", zorder=5)

        lo = min(zt.min(), zp.min()) - 0.02 if len(zt) > 0 else -0.05
        hi = max(zt.max(), zp.max()) + 0.02 if len(zt) > 0 else 0.55
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$z_\mathrm{true}$")
        ax.set_ylabel(r"$z_\mathrm{pred}$")
        ax.text(0.05, 0.94, f"MAE = {mae:.4f}\nn = {len(zt)}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
        ax.grid(True, alpha=0.3, lw=0.6)
        ax.set_title(f"(b{i+1}) {name}", fontsize=11, fontweight="bold")


def panel_benchmark(ax, benchmark_data: dict):
    """Panel C: Grouped bar chart of benchmark scores."""
    if not benchmark_data:
        ax.text(0.5, 0.5, "Benchmark data\nnot yet available",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(c) Benchmark Retention", fontsize=11, fontweight="bold")
        return

    tasks = list(benchmark_data.get("tasks", {}).keys())
    base_scores = [benchmark_data["tasks"][t].get("base", 0) for t in tasks]
    ft_scores = [benchmark_data["tasks"][t].get("finetuned", 0) for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35
    ax.bar(x - width/2, base_scores, width, label="Base", color=COLORS["base"], alpha=0.8)
    ax.bar(x + width/2, ft_scores, width, label="Fine-tuned", color=COLORS["finetuned"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy [%]")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, lw=0.6)
    ax.set_title("(c) Benchmark Retention", fontsize=11, fontweight="bold")


def panel_summary_table(ax, summary: dict):
    """Panel D: Summary metrics as a formatted table."""
    ax.axis("off")
    ax.set_title("(d) Summary", fontsize=11, fontweight="bold")

    rows = []
    for model_name, metrics in summary.items():
        rows.append([
            model_name,
            f"{metrics.get('train_loss_init', '—')}",
            f"{metrics.get('train_loss_final', '—')}",
            f"{metrics.get('mae', '—')}",
            f"{metrics.get('n_valid', '—')}",
            f"{metrics.get('train_time_min', '—')}",
        ])

    col_labels = ["Model", "Loss (init)", "Loss (final)", "MAE", "Valid/N", "Time (min)"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center", colColours=["#e6e6e6"]*len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)


def make_money_plot(runs: dict, eval_dirs: dict, benchmark_data: dict,
                    output_dir: str):
    setup_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel A: loss comparison (top left, spans 1 col)
    ax_loss = fig.add_subplot(gs[0, 0])
    panel_loss_comparison(ax_loss, runs)

    # Panel B: redshift scatter (top middle + right)
    eval_names = list(eval_dirs.keys())
    ax_scatter = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    panel_redshift_scatter(ax_scatter, eval_dirs)

    # Panel C: benchmarks (bottom left)
    ax_bench = fig.add_subplot(gs[1, 0])
    panel_benchmark(ax_bench, benchmark_data)

    # Panel D: summary table (bottom middle + right)
    ax_table = fig.add_subplot(gs[1, 1:])

    # Build summary from available data
    summary = {}
    for name, run_dir in runs.items():
        log = load_trainer_state(Path(run_dir))
        losses = [e["loss"] for e in log if "loss" in e]
        runtime = [e for e in log if "train_runtime" in e]
        entry = {}
        if losses:
            entry["train_loss_init"] = f"{losses[0]:.3f}"
            entry["train_loss_final"] = f"{losses[-1]:.3f}"
        if runtime:
            entry["train_time_min"] = f"{runtime[0]['train_runtime']/60:.0f}"
        # Add eval metrics if available
        for eval_name, eval_dir in eval_dirs.items():
            if name.lower() in eval_name.lower():
                m = load_metrics(Path(eval_dir) / "metrics.json")
                if m:
                    entry["mae"] = f"{m['mae']:.4f}"
                    entry["n_valid"] = f"{m['n_valid']}/{m['n_total']}"
        summary[name] = entry

    panel_summary_table(ax_table, summary)

    save_path = output_dir / "money_plot.png"
    fig.savefig(save_path)
    print(f"Money plot saved to {save_path}")

    # Also save individual panels
    for name, run_dir in runs.items():
        log = load_trainer_state(Path(run_dir))
        if log:
            losses = [e["loss"] for e in log if "loss" in e]
            steps = [e["step"] for e in log if "loss" in e]
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.plot(steps, losses, color=COLORS["primary"], lw=1.5)
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Training Loss")
            ax2.set_title(f"{name} Training Loss")
            ax2.grid(True, alpha=0.3, lw=0.6)
            fig2.tight_layout()
            fig2.savefig(output_dir / f"loss_{name.replace(' ', '_').lower()}.png")

    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="plots/summary")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="name=path pairs for training runs")
    parser.add_argument("--evals", nargs="*", default=None,
                        help="name=path pairs for eval result dirs")
    parser.add_argument("--benchmarks", default=None,
                        help="Path to benchmark comparison JSON")
    args = parser.parse_args()

    # Defaults
    runs = {}
    eval_dirs = {}
    if args.runs:
        for r in args.runs:
            name, path = r.split("=", 1)
            runs[name] = path
    else:
        runs = {
            "GPT-OSS-20B": "output_models/gpt-oss-20b_structured",
            "GPT-OSS-120B": "output_models/gpt-oss-120b_structured",
        }
    if args.evals:
        for e in args.evals:
            name, path = e.split("=", 1)
            eval_dirs[name] = path
    else:
        eval_dirs = {
            "GPT-OSS-20B (FT)": "plots/eval_20b",
            "GPT-OSS-120B (FT)": "plots/eval_120b",
        }

    benchmark_data = {}
    if args.benchmarks and Path(args.benchmarks).exists():
        with open(args.benchmarks) as f:
            benchmark_data = json.load(f)

    make_money_plot(runs, eval_dirs, benchmark_data, args.output_dir)


if __name__ == "__main__":
    main()
