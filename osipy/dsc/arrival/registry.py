"""Bolus arrival time detector registry."""

import logging

import numpy as np

from osipy.common.exceptions import DataValidationError
from osipy.dsc.arrival.base import BaseArrivalDetector

logger = logging.getLogger(__name__)

_ARRIVAL_REGISTRY: dict[str, type] = {}


def register_arrival_detector(name: str):
    """Register a bolus arrival time detection algorithm."""

    def decorator(cls):
        if name in _ARRIVAL_REGISTRY:
            logger.warning("Overwriting arrival detector '%s'", name)
        _ARRIVAL_REGISTRY[name] = cls
        return cls

    return decorator


def get_arrival_detector(name: str):
    """Get an arrival detector instance by name."""
    if name not in _ARRIVAL_REGISTRY:
        valid = ", ".join(sorted(_ARRIVAL_REGISTRY.keys()))
        raise DataValidationError(f"Unknown arrival detector: {name}. Valid: {valid}")
    return _ARRIVAL_REGISTRY[name]()


def list_arrival_detectors() -> list[str]:
    """List registered arrival time detection algorithms."""
    return sorted(_ARRIVAL_REGISTRY.keys())


@register_arrival_detector("residue_peak")
class ResiduePeakDetector(BaseArrivalDetector):
    """Detect arrival time from peak of residue function."""

    @property
    def name(self) -> str:
        """Return human-readable component name."""
        return "Residue Peak"

    @property
    def reference(self) -> str:
        """Return primary literature citation."""
        return "Wu O et al. MRM 2003;50(1):164-174."

    def detect(self, residue_function, dt):
        """Detect arrival time from peak of residue function.

        Parameters
        ----------
        residue_function : NDArray
            Residue function R(t).
        dt : float
            Time step in seconds.

        Returns
        -------
        float
            Estimated arrival time in seconds.
        """
        return float(np.argmax(residue_function) * dt)
