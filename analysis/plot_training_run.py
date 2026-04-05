#!/usr/bin/env python3
"""
plot_training_run.py — Generate plots from a completed HF Trainer run.

Usage:
    python analysis/plot_training_run.py \
        --run_dir output_models/gpt-oss-20b_structured \
        --output_dir plots/gpt-oss-20b_structured
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.plots import setup_style, plot_loss_curves, COLORS

import matplotlib.pyplot as plt


def load_trainer_state(run_dir: Path) -> dict:
    """Find and load trainer_state.json from the latest checkpoint or run root."""
    # Check checkpoints in descending order
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for ckpt in reversed(checkpoints):
        state_file = ckpt / "trainer_state.json"
        if state_file.exists():
            with open(state_file) as f:
                return json.load(f)
    # Fallback to run root
    state_file = run_dir / "trainer_state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    raise FileNotFoundError(f"No trainer_state.json found in {run_dir}")


def plot_loss_and_lr(log_history: list[dict], output_dir: Path):
    """Two-panel figure: loss curve (left) and learning rate schedule (right)."""
    setup_style()

    steps = [e["step"] for e in log_history if "loss" in e]
    losses = [e["loss"] for e in log_history if "loss" in e]
    lrs = [e["learning_rate"] for e in log_history if "learning_rate" in e]
    grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Loss
    ax = axes[0]
    ax.plot(steps, losses, color=COLORS["primary"], lw=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.text(0.95, 0.95, f"Final: {losses[-1]:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

    # Learning rate
    ax = axes[1]
    ax.plot(steps[:len(lrs)], lrs, color=COLORS["secondary"], lw=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-4, -4))
    ax.grid(True, alpha=0.3, lw=0.6)

    # Gradient norm
    ax = axes[2]
    ax.plot(steps[:len(grad_norms)], grad_norms, color=COLORS["accent"], lw=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.grid(True, alpha=0.3, lw=0.6)

    fig.suptitle("gpt-oss-20b LoRA Training (structured verbalization)", fontsize=12, y=1.02)
    fig.tight_layout()
    save_path = output_dir / "training_curves.png"
    fig.savefig(save_path)
    print(f"Saved: {save_path}")
    return fig


def plot_loss_by_epoch(log_history: list[dict], output_dir: Path):
    """Loss curve colored by epoch to show epoch boundary behavior."""
    setup_style()

    steps = [e["step"] for e in log_history if "loss" in e]
    losses = [e["loss"] for e in log_history if "loss" in e]
    epochs = [e["epoch"] for e in log_history if "loss" in e]

    fig, ax = plt.subplots(figsize=(7, 4))

    # Color by epoch
    epoch_nums = np.array(epochs)
    e1_mask = epoch_nums < 1.0
    e2_mask = epoch_nums >= 1.0

    s = np.array(steps)
    l = np.array(losses)

    if e1_mask.any():
        ax.plot(s[e1_mask], l[e1_mask], "o-", color=COLORS["primary"],
                ms=4, lw=1.2, label="Epoch 1", alpha=0.8)
    if e2_mask.any():
        ax.plot(s[e2_mask], l[e2_mask], "o-", color=COLORS["secondary"],
                ms=4, lw=1.2, label="Epoch 2", alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3, lw=0.6)
    fig.tight_layout()
    save_path = output_dir / "loss_by_epoch.png"
    fig.savefig(save_path)
    print(f"Saved: {save_path}")
    return fig


def print_run_summary(state: dict):
    log = state["log_history"]
    losses = [e["loss"] for e in log if "loss" in e]
    print(f"  Total steps:  {state['global_step']}")
    print(f"  Epochs:       {state['num_train_epochs']}")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Min loss:     {min(losses):.4f} (step {[e['step'] for e in log if e.get('loss') == min(losses)][0]})")
    print(f"  Total FLOPs:  {state['total_flos']:.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to training output dir")
    parser.add_argument("--output_dir", default=None, help="Where to save plots (default: plots/<run_name>)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("plots") / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trainer state from {run_dir} ...")
    state = load_trainer_state(run_dir)

    print("Run summary:")
    print_run_summary(state)

    print("\nGenerating plots ...")
    plot_loss_and_lr(state["log_history"], output_dir)
    plot_loss_by_epoch(state["log_history"], output_dir)

    # Also generate the simple loss curve using the existing function
    losses = [e["loss"] for e in state["log_history"] if "loss" in e]
    fig = plot_loss_curves(losses, save_path=output_dir / "loss_curve_simple.png")
    print(f"Saved: {output_dir / 'loss_curve_simple.png'}")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
