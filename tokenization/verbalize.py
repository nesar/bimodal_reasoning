"""
verbalize.py — Convert a galaxy record into a structured text chunk.

The verbalized block masks all physical properties that the model must predict
(redshift, stellar mass, age, metallicity) and produces a structured output
containing all four quantities.

Example input (all targets masked):
    Object: SDSS-sample-004827
    Survey: SDSS, SNR=14.3
    Spectrum: [ <tokens> ]
    Predict: Redshift z, Stellar mass log(M/M☉), Age [Gyr], Metallicity Z

Example output:
    Redshift: z = 0.3214
    Stellar mass: log(M/M☉) = 10.42
    Age: 3.2 Gyr
    Metallicity: Z = 0.021 </s>
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

END_TOKEN = "</s>"


# ---------------------------------------------------------------------------
# Galaxy record
# ---------------------------------------------------------------------------

@dataclass
class GalaxyRecord:
    idx: int
    z: float            # physical redshift
    log_mass: float     # log10(M / M_sun)
    age_gyr: float      # stellar age in Gyr
    metallicity: float  # absolute metallicity Z
    snr: float          # spectrum signal-to-noise ratio
    survey: str = "SDSS"


# ---------------------------------------------------------------------------
# Physical helpers
# ---------------------------------------------------------------------------

def estimate_snr(spec_normalized: np.ndarray) -> float:
    """Proxy SNR from normalized spectrum: peak / noise floor."""
    signal = np.percentile(spec_normalized, 90)
    noise = np.std(spec_normalized - np.median(spec_normalized))
    return float(np.clip(signal / (noise + 1e-8) * 10, 1.0, 99.9))


def object_label(idx: int, survey: str = "SDSS") -> str:
    """Row-index label — the SDSS HDF5 does not contain sky coordinates."""
    return f"{survey}-sample-{idx:06d}"


# ---------------------------------------------------------------------------
# Verbalization
# ---------------------------------------------------------------------------

def verbalize_input(record: GalaxyRecord) -> str:
    """
    Produce the input text block for a galaxy.

    All physical target properties are masked — the model sees only the
    object label, survey metadata, and spectrum tokens.
    """
    return (
        f"Object: {object_label(record.idx, record.survey)}\n"
        f"Survey: {record.survey}, SNR={record.snr:.1f}\n"
        f"Predict: Redshift z, Stellar mass log(M/M☉), Age [Gyr], Metallicity Z"
    )


def verbalize_output(record: GalaxyRecord) -> str:
    """
    Produce the structured output with all four physical properties.
    """
    return (
        f"Redshift: z = {record.z:.4f}\n"
        f"Stellar mass: log(M/M☉) = {record.log_mass:.2f}\n"
        f"Age: {record.age_gyr:.1f} Gyr\n"
        f"Metallicity: Z = {record.metallicity:.3f} {END_TOKEN}"
    )


def verbalize_output_compact(record: GalaxyRecord) -> str:
    """
    Machine-parseable compact output: numbers first, always the same format.

    Format: [z=0.3510|mass=11.18|age=10.4|Z=0.461]
    Regex:  z=([\d.]+)  mass=([\d.]+)  age=([\d.]+)  Z=([\d.]+)
    """
    return (
        f"[z={record.z:.4f}|mass={record.log_mass:.2f}"
        f"|age={record.age_gyr:.1f}|Z={record.metallicity:.3f}]"
    )


# ---------------------------------------------------------------------------
# Fine-tuning pair builder
# ---------------------------------------------------------------------------

def make_ft_pair(record: GalaxyRecord, spectrum_series: Optional[str] = None,
                 compact: bool = False) -> dict:
    """
    Build a {"input": ..., "output": ...} fine-tuning instance.

    Input  = object metadata + spectrum tokens (all targets masked)
    Output = physical properties (compact or verbose format)

    Args:
        compact: if True, use machine-parseable [z=...|mass=...|age=...|Z=...] format
    """
    context = verbalize_input(record)
    if spectrum_series is not None:
        context += f"\nSpectrum: [ {spectrum_series}]"

    output_fn = verbalize_output_compact if compact else verbalize_output
    return {
        "input": context,
        "output": output_fn(record),
    }
