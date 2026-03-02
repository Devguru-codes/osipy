"""Base class for AIF detection algorithms."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from osipy.common.models.base import BaseComponent

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from osipy.common.aif.detection import AIFDetectionParams, AIFDetectionResult
    from osipy.common.dataset import PerfusionDataset


class BaseAIFDetector(BaseComponent):
    """Abstract base class for AIF detection algorithms."""

    @abstractmethod
    def detect(
        self,
        dataset: PerfusionDataset,
        params: AIFDetectionParams | None = None,
        roi_mask: NDArray[np.bool_] | None = None,
    ) -> AIFDetectionResult:
        """Detect AIF from dataset.

        Parameters
        ----------
        dataset : PerfusionDataset
            Input dataset.
        params : AIFDetectionParams | None
            Detection parameters.
        roi_mask : NDArray | None
            ROI mask to restrict search.

        Returns
        -------
        AIFDetectionResult
            Detection result.
        """
        ...
