"""FittableModel protocol and BaseBoundModel base class.

This module defines the interface between signal models and the shared
fitting infrastructure. A ``FittableModel`` is a model with all
independent variables bound (time, AIF, b-values, PLDs, etc.) so that
only free parameters remain.

Each modality provides a wrapper class that creates a ``FittableModel``
from its domain-specific signal model:

- DCE: ``BoundDCEModel(model, t, aif)``
- IVIM: ``BoundIVIMModel(model, b_values)``
- ASL: ``BoundASLModel(plds, m0, params)``
- DSC: ``BoundGammaVariateModel(time)``

The ``BaseBoundModel`` base class handles fixed-parameter logic shared
across all wrappers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from osipy.common.models.base import BaseSignalModel


@runtime_checkable
class FittableModel(Protocol):
    """Protocol for fitting-ready models.

    All independent variables are bound; only free parameters remain.
    The fitter interacts exclusively through this interface and never
    needs to know about modality-specific context (time, AIF, b-values).

    Shapes
    ------
    - ``observed_batch``: ``(n_observations, n_voxels)``
    - ``params_batch``: ``(n_free_params, n_voxels)``
    - ``predict_array_batch`` returns: ``(n_observations, n_voxels)``
    - ``compute_jacobian_batch`` returns: ``(n_free_params, n_observations, n_voxels)`` or ``None``
    """

    @property
    def name(self) -> str:
        """Model display name."""
        ...

    @property
    def parameters(self) -> list[str]:
        """List of free parameter names."""
        ...

    @property
    def parameter_units(self) -> dict[str, str]:
        """Mapping of parameter names to their unit strings."""
        ...

    @property
    def reference(self) -> str:
        """Literature reference for the model."""
        ...

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return lower and upper bounds for each free parameter."""
        ...

    def get_initial_guess_batch(
        self, observed_batch: NDArray[np.floating[Any]], xp: Any
    ) -> NDArray[np.floating[Any]]:
        """Compute initial parameter guesses from observed data.

        Parameters
        ----------
        observed_batch : NDArray
            Observed signal, shape ``(n_observations, n_voxels)``.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray
            Initial guesses, shape ``(n_free_params, n_voxels)``.
        """
        ...

    def predict_array_batch(
        self, params_batch: NDArray[np.floating[Any]], xp: Any
    ) -> NDArray[np.floating[Any]]:
        """Predict signal from parameter values.

        Parameters
        ----------
        params_batch : NDArray
            Parameter values, shape ``(n_free_params, n_voxels)``.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray
            Predicted signal, shape ``(n_observations, n_voxels)``.
        """
        ...

    def compute_jacobian_batch(
        self,
        params_batch: NDArray[np.floating[Any]],
        predicted: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]] | None:
        """Return analytical Jacobian or None for numerical fallback.

        Parameters
        ----------
        params_batch : NDArray
            Parameter values, shape ``(n_free_params, n_voxels)``.
        predicted : NDArray
            Predicted signal, shape ``(n_observations, n_voxels)``.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray | None
            Jacobian of shape ``(n_free_params, n_observations, n_voxels)``,
            or None to use numerical finite differences.
        """
        ...


class BaseBoundModel:
    """Shared base for all modality binding wrappers.

    Handles fixed-parameter logic: filtering parameters, bounds, and
    initial guesses to exclude fixed values, and injecting them back
    into the full parameter array for model evaluation.

    Parameters
    ----------
    model : BaseSignalModel
        The underlying signal model.
    fixed : dict[str, float] | None
        Parameters to fix at constant values during fitting.
        These are removed from the free parameter list and injected
        back into the full array inside ``predict_array_batch``.
    """

    def __init__(
        self, model: BaseSignalModel, fixed: dict[str, float] | None = None
    ) -> None:
        self._model = model
        self._fixed = fixed or {}

        # Compute free params: model.parameters minus fixed keys
        all_params = model.parameters
        self._free_params = [p for p in all_params if p not in self._fixed]

        # Precompute mapping for _expand_params
        self._n_all = len(all_params)
        self._n_free = len(self._free_params)

        # For each position in all_params, record whether it's free or fixed
        # and what value/index to use
        self._free_indices: list[int] = []  # indices into all_params for free params
        self._fixed_indices: list[int] = []  # indices into all_params for fixed params
        self._fixed_values: list[float] = []  # values for fixed params

        for i, p in enumerate(all_params):
            if p in self._fixed:
                self._fixed_indices.append(i)
                self._fixed_values.append(self._fixed[p])
            else:
                self._free_indices.append(i)

    @property
    def name(self) -> str:
        """Model display name, delegated to the wrapped signal model."""
        return self._model.name

    @property
    def parameters(self) -> list[str]:
        """Free parameters only (excludes fixed)."""
        return self._free_params

    @property
    def parameter_units(self) -> dict[str, str]:
        """Unit strings for free parameters only."""
        return {
            k: v
            for k, v in self._model.parameter_units.items()
            if k in self._free_params
        }

    @property
    def reference(self) -> str:
        """Literature reference, delegated to the wrapped signal model."""
        return self._model.reference

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return bounds for free parameters only."""
        all_bounds = self._model.get_bounds()
        return {k: all_bounds[k] for k in self._free_params if k in all_bounds}

    def _expand_params(
        self, free_batch: NDArray[np.floating[Any]], xp: Any
    ) -> NDArray[np.floating[Any]]:
        """Expand free params to full param array by injecting fixed values.

        Parameters
        ----------
        free_batch : NDArray
            Free parameter values, shape ``(n_free, n_voxels)``.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray
            Full parameter array, shape ``(n_all, n_voxels)``.
        """
        if not self._fixed:
            return free_batch

        n_voxels = free_batch.shape[1]
        full = xp.empty((self._n_all, n_voxels), dtype=free_batch.dtype)

        # Place free params at their original positions
        for free_idx, all_idx in enumerate(self._free_indices):
            full[all_idx, :] = free_batch[free_idx, :]

        # Place fixed params
        for fixed_idx, all_idx in enumerate(self._fixed_indices):
            full[all_idx, :] = self._fixed_values[fixed_idx]

        return full

    def ensure_device(self, xp: Any) -> None:
        """Transfer stored arrays to the device associated with *xp*.

        Call this once before the fitting loop so that every internal
        array lives on the same device as the observed data.  The
        method is idempotent — repeated calls with the same *xp* are
        safe and essentially free.

        The base implementation converts any array items in
        ``_fixed_values``.  Subclasses should call ``super()`` and
        then convert their own arrays (time, AIF, b-values, PLDs,
        etc.) via ``xp.asarray()``.

        Parameters
        ----------
        xp : module
            Array module (numpy or cupy).
        """
        self._fixed_values = [
            xp.asarray(v) if hasattr(v, "__array__") else v for v in self._fixed_values
        ]

    def compute_jacobian_batch(
        self,
        params_batch: NDArray[np.floating[Any]],
        predicted: NDArray[np.floating[Any]],
        xp: Any,
    ) -> NDArray[np.floating[Any]] | None:
        """Return analytical Jacobian or None for numerical fallback.

        Default returns None (numerical finite differences).
        Subclasses with analytical Jacobians should override.
        """
        return None
