"""IVIM bi-exponential signal model.

This module implements the IVIM bi-exponential model for separating
diffusion and perfusion contributions in DWI data, following the
OSIPI CAPLEX definitions for IVIM quantities.

IVIM parameters:

- **D** (diffusion coefficient): True tissue diffusion coefficient,
  units mm^2/s.
- **D*** (pseudo-diffusion coefficient): Perfusion-related
  incoherent motion coefficient, units mm^2/s.
- **f** (perfusion fraction): Fraction of signal arising from the
  perfusion (microvascular) compartment, dimensionless [0, 1].

References
----------
.. [1] Le Bihan D et al. (1988). Separation of diffusion and perfusion
   in intravoxel incoherent motion MR imaging. Radiology 168(2):497-505.
.. [2] Lemke A et al. (2010). An in vivo verification of the intravoxel
   incoherent motion effect in diffusion-weighted imaging of the abdomen.
   Magn Reson Med 64(6):1580-1585.
.. [3] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
.. [4] Dickie BR et al. MRM 2024. doi:10.1002/mrm.30101
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from osipy.common.backend.array_module import get_array_module
from osipy.common.exceptions import DataValidationError
from osipy.common.models.base import BaseSignalModel

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class IVIMParams:
    """Parameters for IVIM model.

    All diffusion parameters are stored in SI-consistent units (mm^2/s).

    Attributes
    ----------
    s0 : float
        Signal intensity at b=0 (arbitrary units).
    d : float
        Tissue diffusion coefficient (D) in mm^2/s.
        Typical range: 0.5e-3 to 2.0e-3 mm^2/s.
    d_star : float
        Pseudo-diffusion coefficient (D*) in mm^2/s.
        Represents perfusion contribution.
        Typical range: 5e-3 to 20e-3 mm^2/s (much higher than D).
    f : float
        Perfusion fraction (f), dimensionless [0, 1].
        Fraction of signal from perfusion compartment.
        Typical range: 0.05-0.30.

    References
    ----------
    .. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    """

    s0: float = 1.0
    d: float = 1.0e-3  # mm^2/s
    d_star: float = 10.0e-3  # mm^2/s
    f: float = 0.1


class IVIMModel(BaseSignalModel, ABC):
    """Abstract base class for IVIM models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return model name."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> list[str]:
        """Return parameter names."""
        ...

    def predict(
        self,
        b_values: "NDArray[np.floating[Any]]",
        params: "Any",
        xp: Any = None,
    ) -> "NDArray[np.floating[Any]]":
        """Predict signal at given b-values.

        Accepts both dict/dataclass and array parameters transparently.

        Parameters
        ----------
        b_values : NDArray[np.floating]
            Diffusion weighting values in s/mm².
        params : NDArray or dict or IVIMParams
            Parameter array ``(n_params,)`` / ``(n_params, n_voxels)``,
            or a dict / dataclass that will be converted automatically.
        xp : module, optional
            Array module (numpy or cupy). Inferred from *b_values* when omitted.

        Returns
        -------
        NDArray[np.floating]
            Predicted signal intensity.
        """
        if xp is None:
            xp = get_array_module(b_values)
        if not hasattr(params, "ndim"):
            params = xp.asarray(self.params_to_array(params), dtype=xp.float64)
        return self._predict(b_values, params, xp)

    @abstractmethod
    def _predict(
        self,
        b_values: "NDArray[np.floating[Any]]",
        params: "NDArray[np.floating[Any]]",
        xp: Any,
    ) -> "NDArray[np.floating[Any]]":
        """Core prediction — model implementers override this.

        Uses ``xp`` operations for both single-voxel and batch
        prediction via broadcasting.

        Parameters
        ----------
        b_values : NDArray[np.floating]
            Diffusion weighting values in s/mm².
            Shape ``(n_b,)`` for single voxel,
            ``(n_b, 1)`` for batch.
        params : NDArray[np.floating]
            Parameter array.
            Shape ``(n_params,)`` for single voxel,
            ``(n_params, n_voxels)`` for batch.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Predicted signal intensity.
        """
        ...

    def predict_batch(
        self,
        b_values: "NDArray[np.floating[Any]]",
        params_batch: "NDArray[np.floating[Any]]",
        xp: Any,
    ) -> "NDArray[np.floating[Any]]":
        """Predict signal for multiple voxels simultaneously.

        Reshapes *b_values* to a column vector and delegates to
        :meth:`_predict`, which handles broadcasting.

        Parameters
        ----------
        b_values : NDArray[np.floating]
            b-values in s/mm², shape (n_b,).
        params_batch : NDArray[np.floating]
            Parameter values, shape (n_params, n_voxels).
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Predicted signal, shape (n_b, n_voxels).
        """
        b_col = b_values[:, xp.newaxis]  # (n_b, 1)
        return self._predict(b_col, params_batch, xp)

    @abstractmethod
    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return physiological parameter bounds."""
        ...


class IVIMBiexponentialModel(IVIMModel):
    """IVIM bi-exponential model.

    The bi-exponential model describes the DWI signal as:

        S(b) = S0 * [(1-f) * exp(-b*D) + f * exp(-b*D*)]

    where:
        - S0 is the signal at b=0
        - f is the perfusion fraction
        - D is the tissue diffusion coefficient
        - D* is the pseudo-diffusion coefficient

    This model assumes that D* >> D, so the perfusion component
    decays rapidly at low b-values.

    References
    ----------
    .. [1] Le Bihan D et al. (1988). Radiology 168(2):497-505.
    .. [2] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "IVIM Bi-exponential"

    @property
    def parameters(self) -> list[str]:
        """Return parameter names."""
        return ["S0", "D", "D*", "f"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return parameter units."""
        return {"S0": "a.u.", "D": "mm^2/s", "D*": "mm^2/s", "f": ""}

    @property
    def reference(self) -> str:
        """Return primary literature citation."""
        return "Le Bihan D et al. Radiology 1988;168(2):497-505."

    # Map OSIPI parameter names to IVIMParams dataclass field names.
    _PARAM_FIELD_MAP: ClassVar[dict[str, str]] = {
        "S0": "s0",
        "D": "d",
        "D*": "d_star",
        "f": "f",
    }

    def params_to_array(self, params: Any) -> "NDArray[np.floating[Any]]":
        """Convert IVIMParams or dict to array.

        Handles the ``D*`` → ``d_star`` field mapping.
        """
        if isinstance(params, dict):
            return np.array([params[p] for p in self.parameters])
        return np.array(
            [getattr(params, self._PARAM_FIELD_MAP[p]) for p in self.parameters]
        )

    def _predict(
        self,
        b_values: "NDArray[np.floating[Any]]",
        params: "NDArray[np.floating[Any]]",
        xp: Any,
    ) -> "NDArray[np.floating[Any]]":
        """Predict signal using bi-exponential model.

        Works for both single-voxel and batch via broadcasting.

        Parameters
        ----------
        b_values : NDArray[np.floating]
            b-values, ``(n_b,)`` or ``(n_b, 1)``.
        params : NDArray[np.floating]
            ``[S0, D, D*, f]`` — shape ``(4,)`` or ``(4, n_voxels)``.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Predicted signal.
        """
        s0 = params[0]
        d = params[1]
        d_star = params[2]
        f = params[3]

        # S(b) = S0 × [(1-f) × exp(-b×D) + f × exp(-b×D*)]
        diffusion_term = (1 - f) * xp.exp(-b_values * d)
        perfusion_term = f * xp.exp(-b_values * d_star)

        return s0 * (diffusion_term + perfusion_term)

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return physiological parameter bounds.

        Bounds follow OSIPI TF2.4 recommendations for IVIM parameters.
        f upper bound of 0.7 accommodates tissues with high perfusion
        fraction such as liver and intestine.
        """
        return {
            "S0": (0.0, np.inf),
            "D": (0.1e-3, 5.0e-3),  # mm²/s
            "D*": (2.0e-3, 100.0e-3),  # mm²/s (much higher than D)
            "f": (0.0, 0.7),  # Perfusion fraction (OSIPI allows up to 0.7)
        }

    def get_initial_guess(
        self,
        signal: "NDArray[np.floating[Any]]",
        b_values: "NDArray[np.floating[Any]]",
    ) -> IVIMParams:
        """Compute initial parameter guess.

        Uses a two-step approach:
        1. Fit mono-exponential at high b-values to estimate D
        2. Use ratio at b=0 to estimate f

        Parameters
        ----------
        signal : NDArray[np.floating]
            Measured signal at each b-value.
        b_values : NDArray[np.floating]
            b-values.

        Returns
        -------
        IVIMParams
            Initial parameter estimates.
        """
        xp = get_array_module(signal, b_values)

        # S0 from b=0 signal
        b0_idx = xp.argmin(b_values)
        s0_init = signal[b0_idx]

        # Fit D from high b-values (b > 200)
        high_b_mask = b_values > 200
        if xp.sum(high_b_mask) >= 2:
            # Log-linear fit
            log_signal = xp.log(signal[high_b_mask] / s0_init + 1e-10)
            b_high = b_values[high_b_mask]

            # Linear regression: log(S/S0) = -b × D
            A = xp.stack([b_high, xp.ones_like(b_high)], axis=1)
            result = xp.linalg.lstsq(A, log_signal, rcond=None)
            coeffs = result[0]  # coeffs[0] = slope, coeffs[1] = intercept
            d_init = -coeffs[0]
        else:
            d_init = 1.0e-3

        # Estimate f from ratio
        # At b=0: S(0) = S0
        # At low b: signal drops due to D* component
        low_b_idx = xp.argmin(xp.abs(b_values - 50))  # Find b~50
        if b_values[low_b_idx] > 10:
            # Extrapolate from high-b fit
            s_diffusion_only = s0_init * xp.exp(-b_values[low_b_idx] * d_init)
            f_init = 1 - signal[low_b_idx] / s_diffusion_only
            f_init = xp.clip(f_init, 0.01, 0.5)
        else:
            f_init = 0.1

        # D* is typically 10× D
        d_star_init = 10.0e-3

        return IVIMParams(
            s0=float(s0_init),
            d=float(xp.clip(d_init, 0.5e-3, 3.0e-3)),
            d_star=float(d_star_init),
            f=float(f_init),
        )


class IVIMSimplifiedModel(IVIMModel):
    """Simplified IVIM model.

    Assumes D* >> D, so exp(-b*D*) is approximately 0 for b > b_threshold:

        S(b) = S0 * (1-f) * exp(-b*D)    for b > b_threshold
        S(0) = S0

    This allows two-step fitting:
    1. Fit mono-exponential at high b-values to get D
    2. Fit f from low b-values

    References
    ----------
    .. [1] Le Bihan D et al. (1988). Radiology 168(2):497-505.
    .. [2] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
    """

    def __init__(self, b_threshold: float = 200.0) -> None:
        """Initialize simplified model.

        Parameters
        ----------
        b_threshold : float
            b-value threshold above which perfusion contribution
            is negligible (s/mm²).
        """
        self.b_threshold = b_threshold

    @property
    def name(self) -> str:
        """Return model name."""
        return "IVIM Simplified"

    @property
    def parameters(self) -> list[str]:
        """Return parameter names."""
        return ["S0", "D", "f"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return parameter units."""
        return {"S0": "a.u.", "D": "mm^2/s", "f": ""}

    @property
    def reference(self) -> str:
        """Return primary literature citation."""
        return "Le Bihan D et al. Radiology 1988;168(2):497-505."

    def _predict(
        self,
        b_values: "NDArray[np.floating[Any]]",
        params: "NDArray[np.floating[Any]]",
        xp: Any,
    ) -> "NDArray[np.floating[Any]]":
        """Predict signal using simplified model.

        Works for both single-voxel and batch via broadcasting.

        Parameters
        ----------
        b_values : NDArray[np.floating]
            b-values, ``(n_b,)`` or ``(n_b, 1)``.
        params : NDArray[np.floating]
            ``[S0, D, f]`` — shape ``(3,)`` or ``(3, n_voxels)``.
        xp : module
            Array module (numpy or cupy).

        Returns
        -------
        NDArray[np.floating]
            Predicted signal.
        """
        s0 = params[0]
        d = params[1]
        f = params[2]

        # For high b-values: S(b) = S0 × (1-f) × exp(-b×D)
        # For b=0: S(0) = S0
        signal = xp.where(
            b_values > self.b_threshold,
            s0 * (1 - f) * xp.exp(-b_values * d),
            s0 * ((1 - f) * xp.exp(-b_values * d) + f),
        )

        return signal

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds."""
        return {
            "S0": (0.0, np.inf),
            "D": (0.1e-3, 5.0e-3),
            "f": (0.0, 0.5),
        }


def compute_adc(
    signal: "NDArray[np.floating[Any]]",
    b_values: "NDArray[np.floating[Any]]",
    b_threshold: float = 100.0,
) -> "NDArray[np.floating[Any]]":
    """Compute apparent diffusion coefficient (ADC).

    ADC is computed from mono-exponential fit at high b-values
    where perfusion contribution is minimal.

    Parameters
    ----------
    signal : NDArray[np.floating]
        Signal intensity at each b-value, shape (..., n_b).
    b_values : NDArray[np.floating]
        b-values in s/mm².
    b_threshold : float
        Minimum b-value to include in fit.

    Returns
    -------
    NDArray[np.floating]
        ADC map in mm²/s.
    """
    xp = get_array_module(signal, b_values)

    # Select b-values above threshold
    mask = b_values >= b_threshold
    b_high = b_values[mask]

    if len(b_high) < 2:
        msg = f"Need at least 2 b-values above threshold ({b_threshold})"
        raise DataValidationError(msg)

    # Get signal at selected b-values
    signal_high = signal[..., mask]

    # Normalize by b=0 signal
    b0_idx = xp.argmin(b_values)
    s0 = signal[..., b0_idx : b0_idx + 1]
    s0 = xp.maximum(s0, 1e-10)

    log_ratio = xp.log(signal_high / s0 + 1e-10)

    # Linear fit: log(S/S0) = -b × ADC
    # Solve for each voxel using least squares
    spatial_shape = signal.shape[:-1]

    # Flatten spatial dimensions
    log_ratio_flat = log_ratio.reshape(-1, len(b_high))

    # Design matrix
    X = xp.column_stack([xp.ones(len(b_high)), -b_high])

    # Vectorized batch solve using normal equations
    # X: (n_b, 2) -- same for all voxels
    # log_ratio_flat: (n_voxels, n_b)
    XtX_inv = xp.linalg.inv(X.T @ X)  # (2, 2)
    pseudo_inv = XtX_inv @ X.T  # (2, n_b)
    coeffs_all = pseudo_inv @ log_ratio_flat.T  # (2, n_voxels)
    adc_flat = coeffs_all[1, :]  # ADC is the second coefficient (slope of -b)

    adc = adc_flat.reshape(spatial_shape)

    # Ensure positive ADC
    adc = xp.maximum(adc, 0)

    return adc
