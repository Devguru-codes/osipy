"""PerfusionDataset container for imaging data and metadata.

This module provides the core data container for perfusion imaging data,
supporting 3D and 4D arrays with associated acquisition metadata.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from osipy.common.exceptions import DataValidationError
from osipy.common.types import AcquisitionParams, AnyAcquisitionParams, Modality

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class PerfusionDataset:
    """Container for perfusion imaging data and metadata.

    This dataclass holds the loaded imaging data along with spatial
    transformation, temporal information, and acquisition parameters.

    Attributes
    ----------
    data : NDArray[np.floating]
        3D or 4D array of image data with shape (x, y, z) or (x, y, z, t).
    affine : NDArray[np.floating]
        4x4 affine transformation matrix mapping voxel to world coordinates.
    modality : Modality
        Perfusion imaging modality (DCE, DSC, ASL, IVIM).
    time_points : NDArray[np.floating] | None
        Time vector in seconds for dynamic (4D) data. Must have length
        equal to data.shape[3] if data is 4D.
    acquisition_params : AnyAcquisitionParams
        Modality-specific acquisition parameters.
    source_path : Path | None
        Original file path if loaded from disk.
    source_format : str
        Source file format ('nifti', 'dicom', 'bids').
    quality_mask : NDArray[np.bool_] | None
        Boolean mask of valid voxels. Shape must match spatial dimensions.

    Raises
    ------
    ValueError
        If data is not 3D or 4D, if affine is not 4x4, or if time_points
        length does not match 4th dimension.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.dataset import PerfusionDataset
    >>> from osipy.common.types import Modality, DCEAcquisitionParams
    >>> data = np.random.rand(64, 64, 20, 30)
    >>> affine = np.eye(4)
    >>> time = np.linspace(0, 300, 30)
    >>> params = DCEAcquisitionParams(tr=5.0, te=2.0, flip_angles=[2, 5, 10, 15])
    >>> dataset = PerfusionDataset(
    ...     data=data,
    ...     affine=affine,
    ...     modality=Modality.DCE,
    ...     time_points=time,
    ...     acquisition_params=params,
    ... )
    """

    data: "NDArray[np.floating[Any]]"
    affine: "NDArray[np.floating[Any]]"
    modality: Modality
    time_points: "NDArray[np.floating[Any]] | None" = None
    acquisition_params: AnyAcquisitionParams = field(default_factory=AcquisitionParams)
    source_path: Path | None = None
    source_format: str = "unknown"
    quality_mask: "NDArray[np.bool_] | None" = None

    def __post_init__(self) -> None:
        """Validate dataset after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate data dimensions and consistency.

        Raises
        ------
        ValueError
            If validation fails.
        """
        # Check data dimensions
        if self.data.ndim not in (3, 4):
            msg = f"Data must be 3D or 4D, got {self.data.ndim}D"
            raise DataValidationError(msg)

        # Check affine shape
        if self.affine.shape != (4, 4):
            msg = f"Affine must be 4x4, got {self.affine.shape}"
            raise DataValidationError(msg)

        # Check time_points consistency for 4D data
        if self.data.ndim == 4:
            if self.time_points is None:
                msg = "4D data requires time_points array"
                raise DataValidationError(msg)
            if len(self.time_points) != self.data.shape[3]:
                msg = (
                    f"time_points length ({len(self.time_points)}) must match "
                    f"4th dimension ({self.data.shape[3]})"
                )
                raise DataValidationError(msg)

        # Check quality_mask shape if provided
        if self.quality_mask is not None:
            spatial_shape = self.data.shape[:3]
            if self.quality_mask.shape != spatial_shape:
                msg = (
                    f"quality_mask shape {self.quality_mask.shape} must match "
                    f"spatial dimensions {spatial_shape}"
                )
                raise DataValidationError(msg)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the data array."""
        return self.data.shape

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        """Return the spatial dimensions (x, y, z)."""
        return (self.data.shape[0], self.data.shape[1], self.data.shape[2])

    @property
    def n_timepoints(self) -> int:
        """Return the number of time points (0 for 3D data)."""
        if self.data.ndim == 4:
            return self.data.shape[3]
        return 0

    @property
    def is_dynamic(self) -> bool:
        """Return True if data is 4D (dynamic/time-series)."""
        return self.data.ndim == 4

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        """Return voxel dimensions in mm from affine matrix."""
        return (
            float(np.linalg.norm(self.affine[:3, 0])),
            float(np.linalg.norm(self.affine[:3, 1])),
            float(np.linalg.norm(self.affine[:3, 2])),
        )
