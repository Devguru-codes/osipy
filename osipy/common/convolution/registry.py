"""Convolution method registry."""

import logging
from collections.abc import Callable

from osipy.common.exceptions import DataValidationError

logger = logging.getLogger(__name__)

_CONVOLUTION_REGISTRY: dict[str, Callable] = {}


def register_convolution(name: str):
    """Register a convolution method."""

    def decorator(func: Callable) -> Callable:
        if name in _CONVOLUTION_REGISTRY:
            logger.warning("Overwriting convolution method '%s'", name)
        _CONVOLUTION_REGISTRY[name] = func
        return func

    return decorator


def get_convolution(name: str) -> Callable:
    """Get a convolution method by name."""
    if name not in _CONVOLUTION_REGISTRY:
        valid = ", ".join(sorted(_CONVOLUTION_REGISTRY.keys()))
        raise DataValidationError(f"Unknown convolution method: {name}. Valid: {valid}")
    return _CONVOLUTION_REGISTRY[name]


def list_convolutions() -> list[str]:
    """List registered convolution methods."""
    return sorted(_CONVOLUTION_REGISTRY.keys())
