"""DCE model registry for dynamic lookup.

This module provides the model registry and associated functions
for registering, retrieving, and listing DCE pharmacokinetic models.
"""

import logging
from typing import Any

from osipy.common.exceptions import DataValidationError
from osipy.dce.models.base import BasePerfusionModel

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type[BasePerfusionModel[Any]]] = {}


def register_model(name: str):
    """Decorator to register a DCE pharmacokinetic model.

    Registers a ``BasePerfusionModel`` subclass in ``MODEL_REGISTRY`` so it
    can be looked up by name via ``get_model()`` and used in ``fit_model()``
    and the DCE pipeline.

    Parameters
    ----------
    name : str
        Registry key for the model (e.g. ``'my_model'``).

    Returns
    -------
    Callable
        Class decorator.

    Examples
    --------
    >>> from osipy.dce.models.registry import register_model
    >>> from osipy.dce.models.base import BasePerfusionModel
    >>> @register_model("my_model")
    ... class MyModel(BasePerfusionModel):
    ...     # implement abstract methods ...
    ...     pass
    >>> from osipy.dce.models.registry import get_model
    >>> model = get_model("my_model")
    """

    def decorator(cls: type[BasePerfusionModel[Any]]) -> type[BasePerfusionModel[Any]]:
        if name in MODEL_REGISTRY:
            logger.warning(
                "Overwriting existing model '%s' (%s) with %s",
                name,
                MODEL_REGISTRY[name].__name__,
                cls.__name__,
            )
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str) -> BasePerfusionModel[Any]:
    """Get model instance by name.

    Parameters
    ----------
    name : str
        Model name: 'tofts', 'extended_tofts', 'patlak', '2cxm',
        or any name added via ``register_model()``.

    Returns
    -------
    BasePerfusionModel
        Model instance.

    Raises
    ------
    DataValidationError
        If model name is not recognized.
    """
    if name not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY.keys()))
        msg = f"Unknown model: {name}. Valid models: {valid}"
        raise DataValidationError(msg)
    return MODEL_REGISTRY[name]()


def list_models() -> list[str]:
    """Return names of all registered DCE models.

    Returns
    -------
    list[str]
        Sorted list of registered model names.
    """
    return sorted(MODEL_REGISTRY.keys())
