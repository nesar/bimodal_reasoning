#!/usr/bin/env python3
"""
spec_tokenizer.py — Convert SDSS galaxy spectra to a text2text fine-tuning dataset.

Supports multiple tokenization strategies (see --strategy flag):
  digit_base10          : serialize flux values as comma-separated decimal digits (original)
  digit_base16          : serialize flux values as comma-separated hex digits
  log_scaled            : log-transform spectra before digit serialization
  patch_mean            : downsample spectrum into fixed-length patch means, then serialize
  wavelength_value      : serialize as (wavelength_bin, flux) pairs
  structured_verbalization : rich structured text block (galaxy type, mass, SFR, ...) + spectrum

Output format (text2text.json):
  {
    "type": "text2text",
    "instances": [
      {"input": "Galaxy spectrum ...: [ <tokens> ]",
       "output": "Redshift: [ <value> ] </s>"},
      ...
    ]
  }

Usage:
  python spec_tokenizer.py \\
      --data-path /path/to/sdss_galaxy_spec.hdf5 \\
      --output-dir /path/to/output \\
      --strategy digit_base10 \\
      --num-samples 10000
"""

import argparse
import json
import os
import numpy as np
from dataclasses import dataclass
from functools import partial

# ---------------------------------------------------------------------------
# Core serialization primitives (from original spec_text2text.ipynb)
# ---------------------------------------------------------------------------

def vec_num2repr(val, base, prec, max_val):
    """Convert numbers to a representation in a specified base with given precision."""
    base = float(base)
    sign = 1 * (val >= 0) - 1 * (val < 0)
    val = np.abs(val)
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)).item())

    before_decimals = []
    for i in range(max_bit_pos):
        digit = (val / base ** (max_bit_pos - i - 1)).astype(int)
        before_decimals.append(digit)
        val -= digit * base ** (max_bit_pos - i - 1)
    before_decimals = np.stack(before_decimals, axis=-1)

    if prec > 0:
        after_decimals = []
        for i in range(prec):
            digit = (val / base ** (-i - 1)).astype(int)
            after_decimals.append(digit)
            val -= digit * base ** (-i - 1)
        after_decimals = np.stack(after_decimals, axis=-1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)
    else:
        digits = before_decimals

    return sign, digits


def vec_repr2num(sign, digits, base, prec, half_bin_correction=True):
    """Convert base representation back to numbers."""
    base = float(base)
    bs, D = digits.shape
    digits_flipped = np.flip(digits, axis=-1)
    powers = -np.arange(-prec, -prec + D)
    val = np.sum(digits_flipped / base ** powers, axis=-1)
    if half_bin_correction:
        val += 0.5 / base ** prec
    return sign * val


@dataclass
class SerializerSettings:
    base: int = 10
    prec: int = 2
    signed: bool = True
    fixed_length: bool = False
    max_val: float = 1e7
    time_sep: str = ','
    bit_sep: str = ''
    plus_sign: str = ''
    minus_sign: str = ' -'
    half_bin_correction: bool = True
    decimal_point: str = ''
    missing_str: str = ' Nan'


def serialize_arr(arr, settings: SerializerSettings) -> str:
    """Serialize an array of numbers into a string."""
    assert np.all(np.abs(arr[~np.isnan(arr)]) <= settings.max_val), \
        f"abs(arr) must be <= max_val={settings.max_val}"

    if not settings.signed:
        plus_sign = minus_sign = ''
    else:
        plus_sign = settings.plus_sign
        minus_sign = settings.minus_sign

    vnum2repr = partial(vec_num2repr, base=settings.base, prec=settings.prec, max_val=settings.max_val)
    sign_arr, digits_arr = vnum2repr(np.where(np.isnan(arr), np.zeros_like(arr), arr))
    ismissing = np.isnan(arr)

    def tokenize(arr):
        return ''.join([settings.bit_sep + str(b) for b in arr])

    bit_strs = []
    for sign, digits, missing in zip(sign_arr, digits_arr, ismissing):
        if missing:
            bit_strs.append(settings.missing_str)
            continue
        if not settings.fixed_length:
            nonzero_indices = np.where(digits != 0)[0]
            if len(nonzero_indices) == 0:
                digits = np.array([0])
            else:
                digits = digits[nonzero_indices[0]:]
            prec = settings.prec
            if len(settings.decimal_point):
                digits = np.concatenate([digits[:-prec], np.array([settings.decimal_point]), digits[-prec:]])
        digits = tokenize(digits)
        sign_sep = plus_sign if sign == 1 else minus_sign
        bit_strs.append(sign_sep + digits)

    bit_str = settings.time_sep.join(bit_strs)
    bit_str += settings.time_sep
    return bit_str


# ---------------------------------------------------------------------------
# Tokenization strategies
# ---------------------------------------------------------------------------

END_TOKEN = "</s>"


def strategy_digit_base10(flux_segment, settings=None):
    """Original strategy: serialize flux as base-10 digit tokens."""
    if settings is None:
        settings = SerializerSettings(prec=2, bit_sep="", time_sep=',')
    data = flux_segment * 10
    data = data - np.mean(data)
    return serialize_arr(data, settings)


def strategy_digit_base16(flux_segment, settings=None):
    """Hex encoding: fewer tokens per flux value at the cost of unusual tokens."""
    if settings is None:
        settings = SerializerSettings(base=16, prec=1, bit_sep="", time_sep=',', max_val=200)
    data = flux_segment * 10
    data = data - np.mean(data)
    return serialize_arr(data, settings)


def strategy_log_scaled(flux_segment, settings=None):
    """Log-transform spectra (shift to positive) before serialization."""
    if settings is None:
        settings = SerializerSettings(prec=2, bit_sep="", time_sep=',', signed=False)
    data = flux_segment - flux_segment.min() + 1e-6
    data = np.log1p(data * 10)
    return serialize_arr(data, settings)


def strategy_patch_mean(flux_segment, patch_size=16, settings=None):
    """Downsample by computing mean of non-overlapping patches, then serialize."""
    if settings is None:
        settings = SerializerSettings(prec=2, bit_sep="", time_sep=',')
    n_patches = len(flux_segment) // patch_size
    patches = flux_segment[:n_patches * patch_size].reshape(n_patches, patch_size)
    patch_means = patches.mean(axis=1)
    patch_means = patch_means * 10 - np.mean(patch_means * 10)
    return serialize_arr(patch_means, settings)


def strategy_wavelength_value(flux_segment, wavelength_segment, settings=None):
    """Serialize as interleaved (wavelength_index, flux_value) pairs."""
    if settings is None:
        settings = SerializerSettings(prec=2, bit_sep="", time_sep=',')
    data = flux_segment * 10
    data = data - np.mean(data)
    n = len(data)
    indices = np.arange(n, dtype=float)
    # Interleave: [idx0, val0, idx1, val1, ...]
    interleaved = np.empty(2 * n)
    interleaved[0::2] = indices
    interleaved[1::2] = data
    return serialize_arr(interleaved, settings)


STRATEGY_REGISTRY = {
    "digit_base10": strategy_digit_base10,
    "digit_base16": strategy_digit_base16,
    "log_scaled": strategy_log_scaled,
    "patch_mean": strategy_patch_mean,
    "wavelength_value": strategy_wavelength_value,
    "structured_verbalization": None,   # handled separately in build_dataset
}

STRATEGY_PREFIX = {
    "digit_base10": "Galaxy spectrum is rescaled and encoded to an input series",
    "digit_base16": "Galaxy spectrum hex-encoded series",
    "log_scaled": "Galaxy spectrum log-scaled series",
    "patch_mean": "Galaxy spectrum patch-mean encoded series",
    "wavelength_value": "Galaxy spectrum wavelength-value pair series",
}


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(X_train, y_train, wavelength,
                  strategy="digit_base10",
                  spectrum_length=256,
                  spectrum_stride=8,
                  num_samples=None,
                  num_replica=1,
                  rng_seed=42,
                  y_phys=None):
    """
    Convert X_train spectra and y_train redshifts into text2text instances.

    Args:
        X_train: torch.Tensor (N, 4556)
        y_train: torch.Tensor (N,) or (N, 4) — first column is redshift (normalized)
        wavelength: np.ndarray (4556,)
        strategy: tokenization strategy name
        y_phys: np.ndarray (N, 4) — physical properties [z, age_gyr, metallicity, log_mass]
                Required when strategy="structured_verbalization".

    Returns:
        list of {"input": ..., "output": ...} dicts
    """
    rng = np.random.default_rng(rng_seed)

    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(STRATEGY_REGISTRY.keys())}")

    N = X_train.shape[0]
    if num_samples is not None:
        N = min(N, num_samples)

    X_np = X_train[:N, ::spectrum_stride].numpy()
    y_np = y_train[:N].numpy() if y_train.ndim == 1 else y_train[:N, 0].numpy()
    len_after_stride = X_np.shape[1]

    if len_after_stride < spectrum_length:
        raise ValueError(
            f"After stride {spectrum_stride}, spectrum has {len_after_stride} bins, "
            f"but spectrum_length={spectrum_length} was requested."
        )

    # --- Structured verbalization path ---
    if strategy == "structured_verbalization":
        from verbalize import GalaxyRecord, estimate_snr, make_ft_pair
        if y_phys is None:
            raise ValueError("strategy='structured_verbalization' requires y_phys (physical properties).")
        instances = []
        default_series_fn = strategy_digit_base10   # embed compact spectrum in verbalized block
        for i in range(N):
            flux = X_np[i]
            phys = y_phys[i]                        # [z, age_gyr, metallicity, log_mass]
            record = GalaxyRecord(
                idx=i, z=float(phys[0]), log_mass=float(phys[3]),
                age_gyr=float(phys[1]), metallicity=float(phys[2]),
                snr=estimate_snr(flux),
            )
            for _ in range(num_replica):
                bos = rng.integers(0, len_after_stride - spectrum_length)
                series = default_series_fn(flux[bos: bos + spectrum_length])
                instances.append(make_ft_pair(record, spectrum_series=series))
        rng.shuffle(instances)
        return instances

    # --- All other strategies ---
    tokenize_fn = STRATEGY_REGISTRY[strategy]
    prefix = STRATEGY_PREFIX[strategy]
    instances = []
    for i in range(N):
        flux = X_np[i]
        redshift_int = int(y_np[i] * 10000)
        for _ in range(num_replica):
            bos = rng.integers(0, len_after_stride - spectrum_length)
            segment = flux[bos: bos + spectrum_length]

            if strategy == "wavelength_value":
                wl_segment = wavelength[::spectrum_stride][bos: bos + spectrum_length]
                series = tokenize_fn(segment, wl_segment)
            elif strategy == "patch_mean":
                series = tokenize_fn(segment, patch_size=16)
            else:
                series = tokenize_fn(segment)

            instances.append({
                "input": f"{prefix}: [ {series}]",
                "output": f"Redshift: [ {redshift_int} ] {END_TOKEN}",
            })

    rng.shuffle(instances)
    return instances


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert SDSS spectra to text2text fine-tuning dataset.")
    parser.add_argument("--data-path", required=True, help="Path to sdss_galaxy_spec.hdf5")
    parser.add_argument("--output-dir", required=True, help="Directory for output text2text.json")
    parser.add_argument("--strategy", default="digit_base10",
                        choices=list(STRATEGY_REGISTRY.keys()),
                        help="Tokenization strategy")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit training samples (default: all)")
    parser.add_argument("--spectrum-length", type=int, default=256,
                        help="Number of flux bins per sample")
    parser.add_argument("--spectrum-stride", type=int, default=8,
                        help="Stride for subsampling the 4556-channel spectrum")
    parser.add_argument("--num-replica", type=int, default=1,
                        help="Random crops per spectrum")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print(f"Loading data from {args.data_path} ...")

    y_phys = None
    if args.strategy == "structured_verbalization":
        from data.read_data import read_with_physical
        X_train, _, y_train, y_phys, wavelength = read_with_physical(args.data_path)
    else:
        from data.read_data import read_all
        X_train, _, y_train, _, wavelength = read_all(args.data_path)

    print(f"Building dataset with strategy='{args.strategy}' ...")
    instances = build_dataset(
        X_train, y_train, wavelength,
        strategy=args.strategy,
        spectrum_length=args.spectrum_length,
        spectrum_stride=args.spectrum_stride,
        num_samples=args.num_samples,
        num_replica=args.num_replica,
        rng_seed=args.seed,
        y_phys=y_phys,
    )
    print(f"Generated {len(instances)} instances.")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "text2text.json")
    with open(out_path, "w") as f:
        json.dump({"type": "text2text", "instances": instances}, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
