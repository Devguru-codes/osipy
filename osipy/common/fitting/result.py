"""FittingResult dataclass for model fitting results.

This module provides the data container for storing results from
pharmacokinetic model fitting, including parameters, goodness-of-fit
metrics, and convergence information.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class FittingResult:
    """Results from pharmacokinetic model fitting.

    This dataclass contains the fitted parameters, goodness-of-fit
    metrics, and convergence information for a single voxel or ROI fit.

    Attributes
    ----------
    parameters : dict[str, float]
        Fitted parameter values. Keys are OSIPI-standard parameter names.
    uncertainties : dict[str, float] | None
        Standard errors for each parameter, if computed.
    residuals : NDArray[np.floating]
        Fit residuals (observed - predicted).
    r_squared : float
        Coefficient of determination (R²).
    chi_squared : float | None
        Chi-squared statistic, if weights were provided.
    aic : float | None
        Akaike Information Criterion.
    converged : bool
        Whether the fit converged successfully.
    n_iterations : int
        Number of iterations performed.
    termination_reason : str
        Reason for termination ('converged', 'max_iter', 'tolerance').
    model_name : str
        Name of the model that was fitted.
    initial_guess : dict[str, float]
        Initial parameter values used.
    bounds : dict[str, tuple[float, float]]
        Parameter bounds that were applied.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.fitting.result import FittingResult
    >>> result = FittingResult(
    ...     parameters={"Ktrans": 0.15, "ve": 0.3, "vp": 0.02},
    ...     residuals=np.random.randn(60) * 0.01,
    ...     r_squared=0.95,
    ...     converged=True,
    ...     n_iterations=15,
    ...     termination_reason="converged",
    ...     model_name="ExtendedTofts",
    ...     initial_guess={"Ktrans": 0.1, "ve": 0.2, "vp": 0.01},
    ...     bounds={"Ktrans": (0, 5), "ve": (0, 1), "vp": (0, 0.2)},
    ... )
    """

    # Fitted parameters
    parameters: dict[str, float]
    uncertainties: dict[str, float] | None = None

    # Goodness of fit
    residuals: "NDArray[np.floating[Any]]" = field(default_factory=lambda: np.array([]))
    r_squared: float = 0.0
    chi_squared: float | None = None
    aic: float | None = None

    # Convergence
    converged: bool = False
    n_iterations: int = 0
    termination_reason: str = ""

    # Model info
    model_name: str = ""
    initial_guess: dict[str, float] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Return True if fit converged with reasonable R²."""
        return self.converged and self.r_squared > 0.5

    @property
    def rmse(self) -> float:
        """Return root mean squared error of residuals."""
        if len(self.residuals) == 0:
            return np.nan
        return float(np.sqrt(np.mean(self.residuals**2)))

    @property
    def n_parameters(self) -> int:
        """Return number of fitted parameters."""
        return len(self.parameters)

    def get_parameter(self, name: str, default: float = np.nan) -> float:
        """Get a parameter value by name.

        Parameters
        ----------
        name : str
            Parameter name.
        default : float
            Default value if parameter not found.

        Returns
        -------
        float
            Parameter value.
        """
        return self.parameters.get(name, default)

    def get_uncertainty(self, name: str, default: float = np.nan) -> float:
        """Get uncertainty for a parameter by name.

        Parameters
        ----------
        name : str
            Parameter name.
        default : float
            Default value if uncertainty not available.

        Returns
        -------
        float
            Uncertainty value.
        """
        if self.uncertainties is None:
            return default
        return self.uncertainties.get(name, default)
