"""M0 calibration method registry."""

import logging

from osipy.common.exceptions import DataValidationError

logger = logging.getLogger(__name__)

M0_CALIBRATION_REGISTRY: dict[str, type] = {}


def register_m0_calibration(name: str):
    """Decorator to register an M0 calibration method.

    Parameters
    ----------
    name : str
        Registry key for the method (e.g. ``'voxelwise'``).

    Returns
    -------
    Callable
        Class decorator.
    """

    def decorator(cls):
        if name in M0_CALIBRATION_REGISTRY:
            logger.warning(
                "Overwriting '%s' (%s) with %s",
                name,
                M0_CALIBRATION_REGISTRY[name].__name__,
                cls.__name__,
            )
        M0_CALIBRATION_REGISTRY[name] = cls
        return cls

    return decorator


def get_m0_calibration(name: str):
    """Get an M0 calibration method instance by name.

    Parameters
    ----------
    name : str
        Method name (e.g. ``'voxelwise'``).

    Returns
    -------
    BaseM0Calibration
        Calibration method instance.

    Raises
    ------
    DataValidationError
        If method name is not recognized.
    """
    if name not in M0_CALIBRATION_REGISTRY:
        valid = ", ".join(sorted(M0_CALIBRATION_REGISTRY.keys()))
        raise DataValidationError(f"Unknown M0 calibration: {name}. Valid: {valid}")
    return M0_CALIBRATION_REGISTRY[name]()


def list_m0_calibrations() -> list[str]:
    """Return names of all registered M0 calibration methods.

    Returns
    -------
    list[str]
        Sorted list of registered method names.
    """
    return sorted(M0_CALIBRATION_REGISTRY.keys())
