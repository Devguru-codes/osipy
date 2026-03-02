"""AIF detector registry."""

import logging

from osipy.common.exceptions import DataValidationError

logger = logging.getLogger(__name__)

_AIF_DETECTOR_REGISTRY: dict[str, type] = {}


def register_aif_detector(name: str):
    """Register an AIF detection algorithm."""

    def decorator(cls):
        if name in _AIF_DETECTOR_REGISTRY:
            logger.warning(
                "Overwriting AIF detector '%s' (%s) with %s",
                name,
                _AIF_DETECTOR_REGISTRY[name].__name__,
                cls.__name__,
            )
        _AIF_DETECTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_aif_detector(name: str):
    """Get an AIF detector instance by name."""
    if name not in _AIF_DETECTOR_REGISTRY:
        valid = ", ".join(sorted(_AIF_DETECTOR_REGISTRY.keys()))
        raise DataValidationError(f"Unknown AIF detector: {name}. Valid: {valid}")
    return _AIF_DETECTOR_REGISTRY[name]()


def list_aif_detectors() -> list[str]:
    """List registered AIF detection algorithms."""
    return sorted(_AIF_DETECTOR_REGISTRY.keys())
