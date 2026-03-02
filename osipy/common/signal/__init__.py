"""Signal processing utilities for osipy.

This module provides functions for baseline correction,
temporal filtering, and interpolation.
"""

from osipy.common.signal.baseline import baseline_correction, estimate_baseline_std
from osipy.common.signal.filtering import (
    resample_to_uniform,
    temporal_filter,
    temporal_interpolate,
)

__all__ = [
    "baseline_correction",
    "estimate_baseline_std",
    "resample_to_uniform",
    "temporal_filter",
    "temporal_interpolate",
]
