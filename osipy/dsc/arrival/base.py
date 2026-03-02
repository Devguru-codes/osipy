"""Base class for bolus arrival time detection."""

from abc import abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from osipy.common.models.base import BaseComponent


class BaseArrivalDetector(BaseComponent):
    """Abstract base class for bolus arrival time detection."""

    @abstractmethod
    def detect(
        self,
        residue_function: NDArray[np.floating[Any]],
        dt: float,
    ) -> float:
        """Detect bolus arrival time from residue function.

        Parameters
        ----------
        residue_function : NDArray
            Residue function R(t).
        dt : float
            Time step in seconds.

        Returns
        -------
        float
            Estimated arrival time in seconds.
        """
        ...
