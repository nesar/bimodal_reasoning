#!/usr/bin/env python3
"""
regen_compact.py — Regenerate the structured_verbalization dataset with:
  1. Compact output format: [z=0.3510|mass=11.18|age=10.4|Z=0.461]
  2. Spectrum trimming so input+output fits in block_size tokens

This ensures the model always sees the closing ']' of the spectrum and
learns to immediately produce parseable numeric output.

Usage:
    python tokenization/regen_compact.py \
        --data-path /path/to/sdss_galaxy_spec.hdf5 \
        --output-dir data/datasets/structured_verbalization_compact \
        --block-size 512 \
        --tokenizer-name openai/gpt-oss-20b
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.read_data import read_with_physical
from tokenization.spec_tokenizer import strategy_digit_base10
from tokenization.verbalize import GalaxyRecord, estimate_snr, make_ft_pair


DEFAULT_DATA = "/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/data/Tokyo/Data/sdss_galaxy_spec.hdf5"


def trim_spectrum_for_budget(input_text: str, output_text: str, tokenizer,
                             block_size: int) -> str:
    """Remove flux values from the middle of the spectrum so input+output fits in block_size tokens."""
    full = input_text + "\n" + output_text
    full_len = len(tokenizer(full, truncation=False)["input_ids"])
    if full_len <= block_size:
        return input_text

    spec_start = input_text.find("Spectrum: [")
    spec_end = input_text.rfind("]")
    if spec_start == -1 or spec_end == -1:
        return input_text

    header = input_text[:spec_start + len("Spectrum: [")]
    trailer = input_text[spec_end:]
    flux_str = input_text[spec_start + len("Spectrum: ["):spec_end]
    flux_values = flux_str.split(",")

    lo, hi = 10, len(flux_values)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        half = mid // 2
        trimmed = ",".join(flux_values[:half] + flux_values[-half:])
        candidate = header + trimmed + trailer + "\n" + output_text
        n_tokens = len(tokenizer(candidate, truncation=False)["input_ids"])
        if n_tokens <= block_size:
            lo = mid
        else:
            hi = mid - 1

    half = lo // 2
    trimmed = ",".join(flux_values[:half] + flux_values[-half:])
    return header + trimmed + trailer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", default="data/datasets/structured_verbalization_compact")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--tokenizer-name", default="openai/gpt-oss-20b")
    parser.add_argument("--spectrum-length", type=int, default=256)
    parser.add_argument("--spectrum-stride", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data ...")
    X_train, X_test, y_train_norm, y_phys_train, wavelength = read_with_physical(args.data_path)
    print(f"  Train: {X_train.shape}")

    print(f"Loading tokenizer: {args.tokenizer_name} ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    X_np = X_train[:, ::args.spectrum_stride].numpy()
    rng = np.random.default_rng(42)
    N = X_np.shape[0]
    len_after_stride = X_np.shape[1]

    instances = []
    trimmed_count = 0
    token_lens = []

    print(f"Building {N} compact instances ...")
    for i in range(N):
        flux = X_np[i]
        phys = y_phys_train[i]
        record = GalaxyRecord(
            idx=i, z=float(phys[0]), log_mass=float(phys[3]),
            age_gyr=float(phys[1]), metallicity=float(phys[2]),
            snr=estimate_snr(flux),
        )
        bos = rng.integers(0, len_after_stride - args.spectrum_length)
        series = strategy_digit_base10(flux[bos: bos + args.spectrum_length])
        pair = make_ft_pair(record, spectrum_series=series, compact=True)

        # Trim spectrum to fit in block_size
        input_orig = pair["input"]
        input_trimmed = trim_spectrum_for_budget(
            pair["input"], pair["output"], tokenizer, args.block_size
        )
        if input_trimmed != input_orig:
            trimmed_count += 1
        pair["input"] = input_trimmed

        full = pair["input"] + "\n" + pair["output"]
        token_lens.append(len(tokenizer(full, truncation=False)["input_ids"]))
        instances.append(pair)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{N}] trimmed={trimmed_count}")

    rng.shuffle(instances)

    out_path = os.path.join(args.output_dir, "text2text.json")
    with open(out_path, "w") as f:
        json.dump({"type": "text2text", "instances": instances}, f)

    token_lens = np.array(token_lens)
    print(f"\nDone. {len(instances)} instances saved to {out_path}")
    print(f"  Trimmed spectra: {trimmed_count}/{N} ({100*trimmed_count/N:.0f}%)")
    print(f"  Token lengths: min={token_lens.min()}, max={token_lens.max()}, "
          f"mean={token_lens.mean():.0f}, median={np.median(token_lens):.0f}")
    print(f"  All fit in {args.block_size}: {(token_lens <= args.block_size).all()}")

    # Show examples
    print("\nExample instances:")
    for i in [0, 1, 2]:
        inst = instances[i]
        print(f"  [{i}] input (last 60): ...{inst['input'][-60:]}")
        print(f"      output: {inst['output']}")


if __name__ == "__main__":
    main()
