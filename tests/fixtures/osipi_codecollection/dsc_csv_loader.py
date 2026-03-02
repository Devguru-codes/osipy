"""Load OSIPI CodeCollection DRO CSV files for DSC deconvolution compliance testing.

Parses the dsc_data.csv file from the OSIPI DCE-DSC-MRI CodeCollection
(github.com/OSIPI/DCE-DSC-MRI_CodeCollection) with ground truth CBV and CBF
values for SVD deconvolution benchmarking.

No pandas dependency — uses stdlib csv + numpy.

References
----------
.. [1] van Houdt PJ et al. MRM 2024;91(5):1774-1786. doi:10.1002/mrm.29826
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_CSV_PATH = Path(__file__).parent / "dsc_data.csv"

# OSIPI CodeCollection tolerances from DSCmodels_data.py
_DSC_TOLERANCES = {
    "CBV": {"absolute": 1.0, "relative": 0.1},  # mL/100mL
    "CBF": {"absolute": 15.0, "relative": 0.1},  # mL/100mL/min
}


def load_osipi_dsc_dro() -> list[dict] | None:
    """Load OSIPI DSC DRO CSV data for deconvolution compliance testing.

    Returns
    -------
    list[dict] or None
        List of test case dicts, or None if CSV not found.
        Each dict contains:

        - ``label``: str — test case label
        - ``c_tis``: np.ndarray — tissue concentration curve
        - ``c_aif``: np.ndarray — arterial input function
        - ``cbv``: float — ground truth CBV (mL/100mL)
        - ``cbf``: float — ground truth CBF (mL/100mL/min)
        - ``tr``: float — repetition time (s)
        - ``time``: np.ndarray — time vector (s)
        - ``tolerances``: dict — per-parameter tolerances
    """
    if not _CSV_PATH.exists():
        return None

    with _CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    cases: list[dict] = []
    for row in rows:
        case = _parse_dsc_row(row)
        if case is not None:
            cases.append(case)

    return cases if cases else None


def _parse_dsc_row(row: dict[str, str]) -> dict | None:
    """Parse one DSC CSV row into a test case dict."""
    label = row.get("label", "unknown")

    c_tis = _parse_array(row.get("C_tis", ""))
    c_aif = _parse_array(row.get("C_aif", ""))

    if len(c_tis) == 0 or len(c_aif) == 0:
        return None

    try:
        cbv = float(row["cbv"])
        cbf = float(row["cbf"])
        tr = float(row["tr"])
    except (KeyError, ValueError):
        return None

    n_timepoints = len(c_tis)
    time = np.arange(n_timepoints) * tr

    return {
        "label": label,
        "c_tis": c_tis,
        "c_aif": c_aif,
        "cbv": cbv,
        "cbf": cbf,
        "tr": tr,
        "time": time,
        "tolerances": _DSC_TOLERANCES,
    }


def _parse_array(cell: str) -> np.ndarray:
    """Parse a space-separated float string into a numpy array."""
    if not cell.strip():
        return np.array([])
    return np.fromstring(cell.strip(), sep=" ")
