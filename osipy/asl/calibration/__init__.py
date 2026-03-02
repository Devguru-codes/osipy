"""ASL M0 calibration module.

This module provides functions for M0 calibration in ASL CBF quantification.
M0 (equilibrium magnetization) is expressed in arbitrary units (a.u.) per the
OSIPI ASL Lexicon.

References
----------
.. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
"""

from osipy.asl.calibration.base import BaseM0Calibration
from osipy.asl.calibration.m0 import (
    M0CalibrationParams,
    apply_m0_calibration,
    compute_m0_from_pd,
)
from osipy.asl.calibration.registry import (
    get_m0_calibration,
    list_m0_calibrations,
    register_m0_calibration,
)

__all__ = [
    "BaseM0Calibration",
    "M0CalibrationParams",
    "apply_m0_calibration",
    "compute_m0_from_pd",
    "get_m0_calibration",
    "list_m0_calibrations",
    "register_m0_calibration",
]
