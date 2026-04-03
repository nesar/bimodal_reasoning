"""
plots.py — Publication-quality plotting for all stages of the bimodal_reasoning pipeline.

Usage:
    from analysis.plots import setup_style, plot_redshift_scatter, ...
    setup_style()

Stages covered:
    1. Data characterization   — property distributions, sample spectra
    2. Tokenization            — strategy comparison (token count vs MAE)
    3. Training                — loss curves
    4. Evaluation              — true/predicted scatter, MAE vs hyperparameter,
                                 benchmark comparison (base vs fine-tuned)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

# ── Style ──────────────────────────────────────────────────────────────────

COLORS = {
    "primary":    "#1f77b4",
    "secondary":  "#ff7f0e",
    "accent":     "#2ca02c",
    "neutral":    "#7f7f7f",
    "base":       "#d62728",
    "finetuned":  "#1f77b4",
}

def setup_style():
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.size":          11,
        "axes.labelsize":     12,
        "axes.titlesize":     12,
        "axes.linewidth":     1.1,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size":   5,
        "ytick.major.size":   5,
        "xtick.minor.size":   3,
        "ytick.minor.size":   3,
        "legend.frameon":     True,
        "legend.framealpha":  0.9,
        "legend.fontsize":    10,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })


# ── Density contour helper ─────────────────────────────────────────────────

def _density_contour(ax, x, y, color, levels=(0.68, 0.95)):
    """Overlay 1σ / 2σ KDE contours on ax."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 20:
        return
    kde = gaussian_kde(np.vstack([x, y]))
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xx, yy = np.meshgrid(xi, yi)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    sorted_z = np.sort(zz.ravel())[::-1]
    cumsum = np.cumsum(sorted_z) / sorted_z.sum()
    contour_levels = [sorted_z[np.searchsorted(cumsum, 1 - p)] for p in levels]
    ax.contour(xx, yy, zz, levels=sorted(contour_levels), colors=[color],
               linewidths=0.8, alpha=0.5)
    for i, (lev, frac) in enumerate(zip(contour_levels, levels)):
        ax.contourf(xx, yy, zz, levels=[lev, zz.max()],
                    colors=[color], alpha=0.08 - i * 0.03)


# ── 1. Data characterization ───────────────────────────────────────────────

def plot_data_overview(z, log_mass, age_gyr, metallicity, save_path=None):
    """
    4-panel distribution plot of galaxy physical properties.

    Args:
        z, log_mass, age_gyr, metallicity : 1-D arrays of physical values
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()

    panels = [
        (z,           r"Redshift $z$",            COLORS["primary"]),
        (log_mass,    r"$\log(M_\star/M_\odot)$", COLORS["secondary"]),
        (age_gyr,     r"Age [Gyr]",               COLORS["accent"]),
        (metallicity, r"Metallicity $Z$",          COLORS["neutral"]),
    ]
    for ax, (data, label, color) in zip(axes, panels):
        ax.hist(data, bins=60, color=color, alpha=0.8, edgecolor="none")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3, lw=0.6)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_sample_spectra(wavelength, spectra, redshifts, n=5, save_path=None):
    """
    Plot n example spectra offset vertically, labeled by redshift.

    Args:
        wavelength : (W,) array — wavelength in Å
        spectra    : (N, W) array — normalized flux
        redshifts  : (N,) array — physical redshifts
        n          : number of spectra to show
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n):
        offset = i * 1.2
        ax.plot(wavelength, spectra[i] + offset,
                lw=0.8, color=plt.cm.viridis(i / n),
                label=f"z = {redshifts[i]:.3f}")
    ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
    ax.set_ylabel("Normalized flux (offset)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, lw=0.6)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── 2. Tokenization comparison ─────────────────────────────────────────────

def plot_tokenization_comparison(strategy_names, token_counts, mae_scores,
                                 save_path=None):
    """
    Two-panel figure: token count per sample (left) and redshift MAE (right)
    for each tokenization strategy.

    Args:
        strategy_names : list of str
        token_counts   : list of int — mean tokens per sample
        mae_scores     : list of float — redshift MAE (or None if not yet measured)
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(len(strategy_names))
    width = 0.6

    ax1.bar(x, token_counts, width, color=COLORS["primary"], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategy_names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Mean tokens per sample")
    ax1.grid(True, axis="y", alpha=0.3, lw=0.6)

    colors = [COLORS["accent"] if m is not None else COLORS["neutral"]
              for m in mae_scores]
    heights = [m if m is not None else 0 for m in mae_scores]
    bars = ax2.bar(x, heights, width, color=colors, alpha=0.85)
    for bar, m in zip(bars, mae_scores):
        if m is None:
            ax2.text(bar.get_x() + bar.get_width() / 2, 0.002, "TBD",
                     ha="center", va="bottom", fontsize=8, color="gray")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategy_names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel(r"Redshift MAE $|z_\mathrm{true} - z_\mathrm{pred}|$")
    ax2.grid(True, axis="y", alpha=0.3, lw=0.6)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── 3. Training ────────────────────────────────────────────────────────────

def plot_loss_curves(train_losses, val_losses=None, save_path=None):
    """
    Training (and optional validation) loss vs step.

    Args:
        train_losses : list/array of loss values
        val_losses   : list/array or None
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = np.arange(1, len(train_losses) + 1)
    ax.plot(steps, train_losses, color=COLORS["primary"], lw=1.5, label="Train")
    if val_losses is not None:
        ax.plot(steps, val_losses, color=COLORS["secondary"],
                lw=1.5, linestyle="--", label="Validation")
        ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, lw=0.6)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── 4a. Evaluation — redshift scatter ─────────────────────────────────────

def plot_redshift_scatter(z_true, z_pred, mae=None, color=None,
                          label=None, ax=None, save_path=None):
    """
    Publication-quality true vs predicted redshift scatter plot with KDE contours.

    Args:
        z_true, z_pred : 1-D arrays of physical redshift values
        mae            : float or None (computed internally if None)
        color          : matplotlib color string
        ax             : existing Axes or None (creates new figure)
    """
    setup_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))

    color = color or COLORS["primary"]
    mask = np.isfinite(z_true) & np.isfinite(z_pred)
    zt, zp = z_true[mask], z_pred[mask]
    if mae is None:
        mae = float(np.mean(np.abs(zt - zp)))

    _density_contour(ax, zt, zp, color)
    ax.scatter(zt, zp, s=12, color=color, alpha=0.6,
               edgecolors="none", zorder=5, label=label)

    lo = min(zt.min(), zp.min()) - 0.02
    hi = max(zt.max(), zp.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$z_\mathrm{true}$")
    ax.set_ylabel(r"$z_\mathrm{pred}$")
    ax.text(0.05, 0.94, f"MAE = {mae:.4f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
    ax.grid(True, alpha=0.3, lw=0.6)

    if standalone:
        if save_path:
            fig.savefig(save_path)
        return fig
    return ax


# ── 4b. MAE vs hyperparameter ──────────────────────────────────────────────

def plot_mae_vs_hyperparam(param_values, mae_values, param_name,
                           ax=None, save_path=None):
    """
    Line + scatter of redshift MAE as a function of one hyperparameter.

    Args:
        param_values : list of numeric values (x axis)
        mae_values   : list of float MAE values
        param_name   : axis label string (e.g., "Learning rate", "LoRA rank r")
    """
    setup_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(param_values, mae_values, "o-", color=COLORS["primary"],
            lw=1.5, ms=6)
    ax.set_xlabel(param_name)
    ax.set_ylabel(r"Redshift MAE")
    ax.grid(True, alpha=0.3, lw=0.6)

    if standalone:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path)
        return fig
    return ax


# ── 4c. Benchmark comparison (base vs fine-tuned) ─────────────────────────

def plot_benchmark_comparison(task_names, base_scores, ft_scores,
                              save_path=None):
    """
    Grouped bar chart comparing base model vs fine-tuned model on benchmarks.

    Args:
        task_names  : list of str — benchmark task labels
        base_scores : list of float — accuracy % for base model
        ft_scores   : list of float — accuracy % for fine-tuned model
    """
    setup_style()
    x = np.arange(len(task_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(task_names) * 1.2), 4))

    ax.bar(x - width / 2, base_scores, width, label="Base model",
           color=COLORS["base"], alpha=0.8)
    ax.bar(x + width / 2, ft_scores, width, label="Fine-tuned",
           color=COLORS["finetuned"], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy [%]")
    ax.set_ylim(0, 105)
    ax.axhline(50, color="gray", lw=0.8, linestyle=":", alpha=0.7)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, lw=0.6)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ── 5. Multi-experiment summary ────────────────────────────────────────────

def plot_experiment_summary(results: list[dict], save_path=None):
    """
    4-panel figure showing MAE vs each varied hyperparameter.

    Args:
        results : list of dicts, each with keys:
                  learning_rate, lora_r, num_train_epochs, training_samples,
                  redshift_mae (float)
    """
    import pandas as pd
    setup_style()
    df = pd.DataFrame(results)
    df["redshift_mae"] = pd.to_numeric(df["redshift_mae"], errors="coerce")

    params = [
        ("learning_rate",    "Learning rate"),
        ("lora_r",           "LoRA rank $r$"),
        ("num_train_epochs", "Epochs"),
        ("training_samples", "Training samples"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, (col, label) in zip(axes, params):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        grp = df.dropna(subset=[col, "redshift_mae"]) \
                .groupby(col)["redshift_mae"].mean().reset_index()
        grp = grp.sort_values(col)
        ax.plot(grp[col], grp["redshift_mae"], "o-",
                color=COLORS["primary"], lw=1.5, ms=6)
        ax.set_xlabel(label)
        ax.set_ylabel("Redshift MAE" if ax is axes[0] else "")
        if col == "learning_rate":
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3, lw=0.6)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
