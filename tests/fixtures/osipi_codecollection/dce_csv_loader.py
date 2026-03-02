"""Load OSIPI CodeCollection DRO CSV files for DCE model compliance testing.

Parses externally-generated CSV files from the OSIPI DCE-DSC-MRI CodeCollection
(github.com/OSIPI/DCE-DSC-MRI_CodeCollection) and maps parameter names/units
to osipy conventions.

No pandas dependency — uses stdlib csv + numpy.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from osipy.common.backend.array_module import get_array_module

_CSV_DIR = Path(__file__).parent / "dce" / "csv"

# OSIPI delay tolerance (same for all models)
_DELAY_TOLERANCE = {"absolute": 1.0, "relative": 0.0}  # seconds

# Per-model configuration: filename, CSV group (A or B), parameter mapping,
# and OSIPI-defined tolerances (in osipy unit space).
#
# Group A (Tofts, Extended Tofts): columns t, C, ca, ta — separate AIF time grid
# Group B (Patlak, 2CXM, 2CUM): columns t, C_t, cp_aif — shared time grid
#
# param_columns maps osipy_name -> (csv_column_name, multiplier).
# tolerances maps osipy_name -> {"absolute": float, "relative": float}.
#
# For delay testing:
#   Group A: delay is created programmatically (shift tissue curve in-memory)
#     - Tofts: 10-point shift, Extended Tofts: 5-point shift
#   Group B: separate _delay_5.csv files from OSIPI CodeCollection
_MODEL_CONFIGS: dict[str, dict] = {
    "tofts": {
        "filename": "dce_DRO_data_tofts.csv",
        "group": "A",
        "delay_shift_points": 10,
        "param_columns": {
            "Ktrans": ("Ktrans", 1.0),
            "ve": ("ve", 1.0),
        },
        "tolerances": {
            "Ktrans": {"absolute": 0.005, "relative": 0.1},
            "ve": {"absolute": 0.05, "relative": 0.0},
        },
    },
    "extended_tofts": {
        "filename": "dce_DRO_data_extended_tofts.csv",
        "group": "A",
        "delay_shift_points": 5,
        "param_columns": {
            "Ktrans": ("Ktrans", 1.0),
            "ve": ("ve", 1.0),
            "vp": ("vp", 1.0),
        },
        "tolerances": {
            "Ktrans": {"absolute": 0.005, "relative": 0.1},
            "ve": {"absolute": 0.05, "relative": 0.0},
            "vp": {"absolute": 0.025, "relative": 0.0},
        },
    },
    "patlak": {
        "filename": "patlak_sd_0.02_delay_0.csv",
        "delay_filename": "patlak_sd_0.02_delay_5.csv",
        "group": "B",
        "param_columns": {
            # Patlak PS = Ktrans (both 1/min)
            "Ktrans": ("ps", 1.0),
            "vp": ("vp", 1.0),
        },
        "tolerances": {
            "Ktrans": {"absolute": 0.005, "relative": 0.1},
            "vp": {"absolute": 0.025, "relative": 0.0},
        },
    },
    "2cxm": {
        "filename": "2cxm_sd_0.001_delay_0.csv",
        "delay_filename": "2cxm_sd_0.001_delay_5.csv",
        "group": "B",
        "param_columns": {
            "Fp": ("fp", 1.0),
            "PS": ("ps", 100.0),  # CSV 1/min -> osipy mL/min/100mL
            "ve": ("ve", 1.0),
            "vp": ("vp", 1.0),
        },
        "tolerances": {
            "Fp": {"absolute": 5.0, "relative": 0.1},
            "PS": {
                "absolute": 0.5,
                "relative": 0.15,
            },  # relaxed: 4-param identifiability
            "ve": {"absolute": 0.05, "relative": 0.1},
            "vp": {"absolute": 0.025, "relative": 0.1},
        },
    },
    "2cum": {
        "filename": "2cum_sd_0.0025_delay_0.csv",
        "delay_filename": "2cum_sd_0.0025_delay_5.csv",
        "group": "B",
        "param_columns": {
            "Fp": ("fp", 1.0),
            "PS": ("ps", 100.0),  # CSV 1/min -> osipy mL/min/100mL
            "vp": ("vp", 1.0),
        },
        "tolerances": {
            "Fp": {"absolute": 5.0, "relative": 0.1},
            "PS": {"absolute": 0.5, "relative": 0.1},  # 0.005 * 100
            "vp": {"absolute": 0.025, "relative": 0.0},
        },
    },
}


def load_osipi_dro(model_name: str) -> list[dict] | None:
    """Load real OSIPI DRO CSV data for a DCE model.

    Parameters
    ----------
    model_name : str
        Model registry name (e.g., ``"tofts"``, ``"2cxm"``).

    Returns
    -------
    list[dict] or None
        List of test case dicts, or None if CSV not found.
        Each dict contains:

        - ``label``: str — test case label from CSV
        - ``time``: np.ndarray — time grid in seconds
        - ``aif``: np.ndarray — AIF on tissue time grid (mM)
        - ``concentration``: np.ndarray — noisy tissue concentration (mM)
        - ``true_params``: dict[str, float] — ground-truth params in osipy units
        - ``tolerances``: dict[str, dict[str, float]] — per-param atol/rtol
    """
    config = _MODEL_CONFIGS.get(model_name)
    if config is None:
        return None

    csv_path = _CSV_DIR / config["filename"]
    if not csv_path.exists():
        return None

    rows = _read_csv(csv_path)
    if not rows:
        return None

    cases: list[dict] = []
    for row in rows:
        case = _parse_row(row, config)
        if case is not None:
            cases.append(case)

    return cases if cases else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV into list of row dicts."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _parse_row(row: dict[str, str], config: dict) -> dict | None:
    """Parse one CSV row into a test case dict."""
    group = config["group"]
    param_columns = config["param_columns"]
    tolerances = config["tolerances"]

    label = row.get("label", "unknown")

    # Parse time and concentration arrays
    if group == "A":
        time = _parse_array(row["t"])
        concentration = _parse_array(row["C"])
        aif_raw = _parse_array(row["ca"])
        ta = _parse_array(row["ta"])

        # Interpolate AIF onto tissue time grid if grids differ
        if len(ta) != len(time) or not np.allclose(ta, time):
            aif = np.interp(time, ta, aif_raw)
        else:
            aif = aif_raw
    elif group == "B":
        time = _parse_array(row["t"])
        concentration = _parse_array(row["C_t"])
        aif = _parse_array(row["cp_aif"])
    else:
        return None

    if len(time) == 0 or len(concentration) == 0 or len(aif) == 0:
        return None

    # Extract ground-truth parameters with unit conversion
    true_params: dict[str, float] = {}
    for osipy_name, (csv_col, multiplier) in param_columns.items():
        raw_val = row.get(csv_col)
        if raw_val is None:
            return None
        true_params[osipy_name] = float(raw_val) * multiplier

    return {
        "label": label,
        "time": time,
        "aif": aif,
        "concentration": concentration,
        "true_params": true_params,
        "tolerances": tolerances,
    }


def load_osipi_dro_delay(model_name: str) -> list[dict] | None:
    """Load OSIPI DRO data with arterial delay for delay recovery testing.

    For Group B models (Patlak, 2CXM, 2CUM), loads the separate ``_delay_5.csv``
    files from the OSIPI CodeCollection.

    For Group A models (Tofts, Extended Tofts), programmatically creates delayed
    tissue curves from the base CSV by prepending zeros and truncating, matching
    the OSIPI CodeCollection ``DCEmodels_data.py`` approach.

    Parameters
    ----------
    model_name : str
        Model registry name (e.g., ``"tofts"``, ``"2cxm"``).

    Returns
    -------
    list[dict] or None
        List of test case dicts, or None if data not available.
        Each dict contains the same keys as ``load_osipi_dro`` plus
        ``"delay"`` in ``true_params`` (ground truth delay in seconds).
    """
    config = _MODEL_CONFIGS.get(model_name)
    if config is None:
        return None

    group = config["group"]

    if group == "B":
        return _load_delay_group_b(config)
    elif group == "A":
        return _load_delay_group_a(config)
    return None


def _load_delay_group_b(config: dict) -> list[dict] | None:
    """Load delay cases from separate delay CSV files (Group B models)."""
    delay_filename = config.get("delay_filename")
    if delay_filename is None:
        return None

    csv_path = _CSV_DIR / delay_filename
    if not csv_path.exists():
        return None

    rows = _read_csv(csv_path)
    if not rows:
        return None

    tolerances = {**config["tolerances"], "delay": _DELAY_TOLERANCE}

    cases: list[dict] = []
    for row in rows:
        case = _parse_row(row, config)
        if case is None:
            continue

        # Read the ground truth delay from CSV
        delay_val = row.get("arterial_delay")
        if delay_val is None:
            continue
        case["true_params"]["delay"] = float(delay_val)
        case["tolerances"] = tolerances
        cases.append(case)

    return cases if cases else None


def _load_delay_group_a(config: dict) -> list[dict] | None:
    """Create delayed cases programmatically from Group A CSV data.

    Matches the OSIPI CodeCollection ``DCEmodels_data.py`` approach:
    shift the tissue concentration curve by N time points (prepend zeros,
    truncate end) and adjust the ground truth delay accordingly.
    """
    csv_path = _CSV_DIR / config["filename"]
    if not csv_path.exists():
        return None

    rows = _read_csv(csv_path)
    if not rows:
        return None

    shift_points = config.get("delay_shift_points", 0)
    if shift_points == 0:
        return None

    tolerances = {**config["tolerances"], "delay": _DELAY_TOLERANCE}

    cases: list[dict] = []
    for row in rows:
        case = _parse_row(row, config)
        if case is None:
            continue

        concentration = case["concentration"]
        time = case["time"]

        # Shift tissue curve: remove last N points, prepend N zeros
        xp = get_array_module(concentration)
        shifted = xp.concatenate(
            [
                xp.zeros(shift_points),
                concentration[:-shift_points],
            ]
        )
        case["concentration"] = shifted

        # Ground truth delay = time value at the shift offset
        case["true_params"]["delay"] = float(time[shift_points])
        case["tolerances"] = tolerances
        case["label"] = case["label"] + "_delayed"
        cases.append(case)

    return cases if cases else None


def _parse_array(cell: str) -> np.ndarray:
    """Parse a space-separated float string into a numpy array."""
    return np.fromstring(cell.strip(), sep=" ")
