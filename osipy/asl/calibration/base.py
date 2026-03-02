"""Base class for M0 calibration methods."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from osipy.common.models.base import BaseComponent

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class BaseM0Calibration(BaseComponent):
    """Abstract base class for M0 calibration strategies."""

    @abstractmethod
    def calibrate(
        self,
        asl_data: "NDArray[np.floating[Any]]",
        m0_image: "NDArray[np.floating[Any]]",
        params: Any,
        mask: "NDArray[np.bool_] | None" = None,
    ) -> tuple["NDArray[np.floating[Any]]", "NDArray[np.floating[Any]]"]:
        """Apply M0 calibration. Returns (calibrated_data, m0_values)."""
        ...
