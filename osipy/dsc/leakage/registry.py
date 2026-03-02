"""Leakage correction method registry."""

import logging

from osipy.common.exceptions import DataValidationError

logger = logging.getLogger(__name__)

LEAKAGE_REGISTRY: dict[str, type] = {}


def register_leakage_corrector(name: str):
    """Register a leakage correction method class."""

    def decorator(cls):
        if name in LEAKAGE_REGISTRY:
            logger.warning(
                "Overwriting '%s' (%s) with %s",
                name,
                LEAKAGE_REGISTRY[name].__name__,
                cls.__name__,
            )
        LEAKAGE_REGISTRY[name] = cls
        return cls

    return decorator


def get_leakage_corrector(name: str):
    """Get a leakage corrector instance by name."""
    if name not in LEAKAGE_REGISTRY:
        valid = ", ".join(sorted(LEAKAGE_REGISTRY.keys()))
        raise DataValidationError(f"Unknown leakage corrector: {name}. Valid: {valid}")
    return LEAKAGE_REGISTRY[name]()


def list_leakage_correctors() -> list[str]:
    """List registered leakage correction methods."""
    return sorted(LEAKAGE_REGISTRY.keys())
