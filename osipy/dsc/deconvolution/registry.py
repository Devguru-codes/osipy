"""Deconvolution method registry."""

import logging

from osipy.common.exceptions import DataValidationError

logger = logging.getLogger(__name__)

DECONVOLUTION_REGISTRY: dict[str, type] = {}


def register_deconvolver(name: str):
    """Register a deconvolution method class."""

    def decorator(cls):
        if name in DECONVOLUTION_REGISTRY:
            logger.warning(
                "Overwriting '%s' (%s) with %s",
                name,
                DECONVOLUTION_REGISTRY[name].__name__,
                cls.__name__,
            )
        DECONVOLUTION_REGISTRY[name] = cls
        return cls

    return decorator


def get_deconvolver(name: str):
    """Get a deconvolver instance by name."""
    if name not in DECONVOLUTION_REGISTRY:
        valid = ", ".join(sorted(DECONVOLUTION_REGISTRY.keys()))
        raise DataValidationError(f"Unknown deconvolver: {name}. Valid: {valid}")
    return DECONVOLUTION_REGISTRY[name]()


def list_deconvolvers() -> list[str]:
    """List registered deconvolution methods."""
    return sorted(DECONVOLUTION_REGISTRY.keys())
