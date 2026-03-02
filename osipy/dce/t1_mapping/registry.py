"""T1 method registry."""

import logging
from collections.abc import Callable

from osipy.common.exceptions import DataValidationError

logger = logging.getLogger(__name__)

_T1_METHOD_REGISTRY: dict[str, Callable] = {}


def register_t1_method(name: str):
    """Decorator to register a T1 mapping method."""

    def decorator(func: Callable) -> Callable:
        if name in _T1_METHOD_REGISTRY:
            logger.warning("Overwriting T1 method '%s'", name)
        _T1_METHOD_REGISTRY[name] = func
        return func

    return decorator


def get_t1_method(name: str) -> Callable:
    """Get a T1 mapping method by name."""
    if name not in _T1_METHOD_REGISTRY:
        valid = ", ".join(sorted(_T1_METHOD_REGISTRY.keys()))
        raise DataValidationError(f"Unknown T1 method: {name}. Valid: {valid}")
    return _T1_METHOD_REGISTRY[name]


def list_t1_methods() -> list[str]:
    """List registered T1 mapping methods."""
    return sorted(_T1_METHOD_REGISTRY.keys())
