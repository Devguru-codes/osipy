"""OSIPI CodeCollection reference data for cross-implementation testing."""

from __future__ import annotations

import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent


def load_reference(model_name: str) -> dict | None:
    """Load OSIPI CodeCollection reference data for a model.

    Parameters
    ----------
    model_name : str
        Model name (e.g., 'tofts', 'extended_tofts', '2cum').

    Returns
    -------
    dict or None
        Reference data dict, or None if file doesn't exist.
    """
    path = _DATA_DIR / "dce" / f"{model_name}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)
