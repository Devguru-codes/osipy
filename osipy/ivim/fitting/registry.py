"""IVIM fitting strategy registry."""

import logging
from collections.abc import Callable

from osipy.common.exceptions import DataValidationError

logger = logging.getLogger(__name__)

_IVIM_FITTER_REGISTRY: dict[str, Callable] = {}


def register_ivim_fitter(name: str):
    """Register an IVIM fitting strategy."""

    def decorator(func: Callable) -> Callable:
        if name in _IVIM_FITTER_REGISTRY:
            logger.warning("Overwriting IVIM fitter '%s'", name)
        _IVIM_FITTER_REGISTRY[name] = func
        return func

    return decorator


def get_ivim_fitter(name: str) -> Callable:
    """Get an IVIM fitting strategy by name."""
    if name not in _IVIM_FITTER_REGISTRY:
        valid = ", ".join(sorted(_IVIM_FITTER_REGISTRY.keys()))
        raise DataValidationError(f"Unknown IVIM fitter: {name}. Valid: {valid}")
    return _IVIM_FITTER_REGISTRY[name]


def list_ivim_fitters() -> list[str]:
    """List registered IVIM fitting strategies."""
    return sorted(_IVIM_FITTER_REGISTRY.keys())
