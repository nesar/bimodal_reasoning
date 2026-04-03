"""
run_tokenize.py — Run digit_base10 tokenization and produce diagnostic plots.

Usage:
    python tokenization/run_tokenize.py --data-path /path/to/sdss_galaxy_spec.hdf5
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.read_data import read_with_physical
from tokenization.spec_tokenizer import build_dataset, strategy_digit_base10, SerializerSettings
from tokenization.verbalize import GalaxyRecord, estimate_snr, make_ft_pair
from analysis.plots import (
    setup_style, plot_data_overview, plot_sample_spectra,
    plot_tokenization_comparison,
)
import matplotlib.pyplot as plt

DATA_PATH = "/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/data/Tokyo/Data/sdss_galaxy_spec.hdf5"
OUT_DIR   = "/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/bimodal_reasoning/pub_results"
DS_DIR    = "/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/bimodal_reasoning/data/datasets/spec_text2text"


def token_count(instance):
    """Count comma-separated tokens in the spectrum series."""
    series = instance["input"].split("[ ")[1].split("]")[0]
    return len([t for t in series.split(",") if t.strip()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DATA_PATH)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DS_DIR, exist_ok=True)
    setup_style()

    # ── Load ──────────────────────────────────────────────────────────
    print("Loading data ...")
    X_train, X_test, y_train_norm, y_phys_train, wavelength = read_with_physical(args.data_path)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # y_phys columns: [z, age_gyr, metallicity, log_mass]
    z        = y_phys_train[:, 0]
    age_gyr  = y_phys_train[:, 1]
    met      = y_phys_train[:, 2]
    log_mass = y_phys_train[:, 3]

    # ── Plot 1: data overview ─────────────────────────────────────────
    print("Plotting data overview ...")
    fig = plot_data_overview(z, log_mass, age_gyr, met,
                             save_path=f"{OUT_DIR}/data_overview.png")
    plt.close(fig)

    # ── Plot 2: sample spectra ────────────────────────────────────────
    print("Plotting sample spectra ...")
    specs_np = X_train[:50].numpy()           # use first 50 for selection
    # pick 6 spread across redshift range
    idx_sorted = np.argsort(z[:50])
    picks = idx_sorted[np.linspace(0, 49, 6, dtype=int)]
    fig = plot_sample_spectra(wavelength, specs_np[picks], z[picks], n=6,
                              save_path=f"{OUT_DIR}/sample_spectra.png")
    plt.close(fig)

    # ── Tokenize ──────────────────────────────────────────────────────
    print("Tokenizing (digit_base10, 2939 train samples) ...")
    instances = build_dataset(
        X_train, y_train_norm, wavelength,
        strategy="digit_base10",
        spectrum_length=256,
        spectrum_stride=8,
        num_samples=None,
        num_replica=1,
        rng_seed=42,
    )
    print(f"  Generated {len(instances)} instances")

    with open(f"{DS_DIR}/text2text.json", "w") as f:
        json.dump({"type": "text2text", "instances": instances}, f)
    print(f"  Saved to {DS_DIR}/text2text.json")

    # ── Plot 3: flux token value distribution ────────────────────────
    # Each spectrum encodes 256 flux values as integers.
    # We want the distribution of those integer values across the full dataset.
    print("Computing flux token value distributions ...")
    all_values = []
    for inst in instances:
        series_str = inst["input"].split("[ ")[1].split("]")[0]
        vals = [int(t.strip()) for t in series_str.split(",") if t.strip().lstrip("-").isdigit()]
        all_values.extend(vals)
    all_values = np.array(all_values)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    axes[0].hist(all_values, bins=80, color="#1f77b4", alpha=0.85, edgecolor="none")
    axes[0].set_xlabel("Flux token value (integer)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(0, color="k", lw=0.8, linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3, lw=0.6)

    # Cumulative: what fraction of values fit within [-k, k]?
    ks = np.arange(0, 200)
    frac = np.array([np.mean(np.abs(all_values) <= k) for k in ks])
    axes[1].plot(ks, frac * 100, color="#1f77b4", lw=1.8)
    axes[1].axhline(95, color="k", lw=0.8, linestyle="--", alpha=0.5, label="95%")
    axes[1].axhline(99, color="gray", lw=0.8, linestyle=":", alpha=0.7, label="99%")
    k95 = ks[np.searchsorted(frac, 0.95)]
    axes[1].axvline(k95, color="#ff7f0e", lw=1.2, linestyle="--", label=f"|v| ≤ {k95} covers 95%")
    axes[1].set_xlabel("Absolute flux token value $|v|$")
    axes[1].set_ylabel("Cumulative coverage [%]")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, lw=0.6)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/flux_token_dist.png")
    plt.close(fig)

    # ── Plot 4: example instance (structured verbalization) ─────────────
    print("Plotting example structured verbalization instance ...")
    # Build one structured instance from the first training sample
    flux_ex = X_train[0, ::8].numpy()
    phys_ex = y_phys_train[0]  # [z, age_gyr, metallicity, log_mass]
    record_ex = GalaxyRecord(
        idx=0, z=float(phys_ex[0]), log_mass=float(phys_ex[3]),
        age_gyr=float(phys_ex[1]), metallicity=float(phys_ex[2]),
        snr=estimate_snr(flux_ex),
    )
    series_ex = strategy_digit_base10(flux_ex[:256])
    ex = make_ft_pair(record_ex, spectrum_series=series_ex)

    # Split input into metadata lines and spectrum token preview
    input_lines = ex["input"].split("\n")
    meta_lines = [l for l in input_lines if not l.startswith("Spectrum:")]
    spec_line = next((l for l in input_lines if l.startswith("Spectrum:")), "")
    if spec_line:
        series_tokens = spec_line.split("[ ")[1].rstrip("]").strip()
        token_list = [t.strip() for t in series_tokens.split(",") if t.strip()]
        token_preview = "Spectrum: [ " + ", ".join(token_list[:30]) + f",  ... ({len(token_list)} total) ]"
    else:
        token_preview = ""

    fig, axes = plt.subplots(2, 1, figsize=(10, 5),
                             gridspec_kw={"height_ratios": [1.3, 1]})
    for ax in axes:
        ax.axis("off")

    axes[0].text(0.0, 1.0, "INPUT", transform=axes[0].transAxes,
                 fontsize=9, fontweight="bold", va="top")
    axes[0].text(0.0, 0.85, "\n".join(meta_lines), transform=axes[0].transAxes,
                 fontsize=8, va="top", family="monospace", color="#333333")
    axes[0].text(0.0, 0.25, token_preview, transform=axes[0].transAxes,
                 fontsize=7.5, va="top", family="monospace", color="#1f77b4",
                 wrap=True)

    axes[1].text(0.0, 1.0, "OUTPUT", transform=axes[1].transAxes,
                 fontsize=9, fontweight="bold", va="top")
    axes[1].text(0.0, 0.8, ex["output"], transform=axes[1].transAxes,
                 fontsize=9, va="top", family="monospace", color="#2ca02c")

    fig.tight_layout(pad=1.2)
    fig.savefig(f"{OUT_DIR}/example_instance.png")
    plt.close(fig)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\nDone. Plots saved to {OUT_DIR}/")
    print(f"  data_overview.png")
    print(f"  sample_spectra.png")
    print(f"  flux_token_dist.png")
    print(f"  example_instance.png")
    print(f"\nDataset stats:")
    print(f"  {len(instances)} instances  |  256 flux bins/sample")
    print(f"  Flux token values: mean={all_values.mean():.1f}  "
          f"std={all_values.std():.1f}  range=[{all_values.min()}, {all_values.max()}]")


if __name__ == "__main__":
    main()
