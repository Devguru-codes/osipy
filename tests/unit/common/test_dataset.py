"""Unit tests for osipy.common.dataset module."""

import numpy as np
import pytest

from osipy.common.dataset import PerfusionDataset
from osipy.common.exceptions import DataValidationError
from osipy.common.types import DCEAcquisitionParams, Modality


class TestPerfusionDataset:
    """Tests for PerfusionDataset dataclass."""

    @pytest.fixture
    def sample_3d_data(self) -> np.ndarray:
        """Create sample 3D data."""
        return np.random.rand(64, 64, 20).astype(np.float64)

    @pytest.fixture
    def sample_4d_data(self) -> np.ndarray:
        """Create sample 4D data."""
        return np.random.rand(64, 64, 20, 30).astype(np.float64)

    @pytest.fixture
    def sample_affine(self) -> np.ndarray:
        """Create sample affine matrix."""
        affine = np.eye(4)
        affine[0, 0] = 2.0  # 2mm voxel size
        affine[1, 1] = 2.0
        affine[2, 2] = 3.0  # 3mm slice thickness
        return affine

    def test_3d_dataset_creation(
        self, sample_3d_data: np.ndarray, sample_affine: np.ndarray
    ) -> None:
        """Test creation of 3D dataset."""
        dataset = PerfusionDataset(
            data=sample_3d_data,
            affine=sample_affine,
            modality=Modality.DCE,
        )
        assert dataset.shape == (64, 64, 20)
        assert dataset.is_dynamic is False
        assert dataset.n_timepoints == 0

    def test_4d_dataset_creation(
        self, sample_4d_data: np.ndarray, sample_affine: np.ndarray
    ) -> None:
        """Test creation of 4D dataset."""
        time_points = np.linspace(0, 300, 30)
        dataset = PerfusionDataset(
            data=sample_4d_data,
            affine=sample_affine,
            modality=Modality.DCE,
            time_points=time_points,
        )
        assert dataset.shape == (64, 64, 20, 30)
        assert dataset.is_dynamic is True
        assert dataset.n_timepoints == 30
        assert dataset.spatial_shape == (64, 64, 20)

    def test_voxel_size(
        self, sample_3d_data: np.ndarray, sample_affine: np.ndarray
    ) -> None:
        """Test voxel size extraction from affine."""
        dataset = PerfusionDataset(
            data=sample_3d_data,
            affine=sample_affine,
            modality=Modality.DCE,
        )
        voxel_size = dataset.voxel_size
        assert voxel_size[0] == pytest.approx(2.0)
        assert voxel_size[1] == pytest.approx(2.0)
        assert voxel_size[2] == pytest.approx(3.0)

    def test_invalid_data_dimensions(self, sample_affine: np.ndarray) -> None:
        """Test that 2D data raises ValueError."""
        data_2d = np.random.rand(64, 64)
        with pytest.raises(DataValidationError, match="must be 3D or 4D"):
            PerfusionDataset(
                data=data_2d,
                affine=sample_affine,
                modality=Modality.DCE,
            )

    def test_invalid_affine_shape(self, sample_3d_data: np.ndarray) -> None:
        """Test that non-4x4 affine raises ValueError."""
        bad_affine = np.eye(3)
        with pytest.raises(DataValidationError, match="must be 4x4"):
            PerfusionDataset(
                data=sample_3d_data,
                affine=bad_affine,
                modality=Modality.DCE,
            )

    def test_4d_data_requires_time_points(
        self, sample_4d_data: np.ndarray, sample_affine: np.ndarray
    ) -> None:
        """Test that 4D data requires time_points."""
        with pytest.raises(DataValidationError, match="requires time_points"):
            PerfusionDataset(
                data=sample_4d_data,
                affine=sample_affine,
                modality=Modality.DCE,
                time_points=None,
            )

    def test_time_points_length_mismatch(
        self, sample_4d_data: np.ndarray, sample_affine: np.ndarray
    ) -> None:
        """Test time_points length validation."""
        time_points = np.linspace(0, 300, 20)  # Wrong length
        with pytest.raises(DataValidationError, match="must match"):
            PerfusionDataset(
                data=sample_4d_data,
                affine=sample_affine,
                modality=Modality.DCE,
                time_points=time_points,
            )

    def test_quality_mask_shape_validation(
        self, sample_3d_data: np.ndarray, sample_affine: np.ndarray
    ) -> None:
        """Test quality_mask shape validation."""
        bad_mask = np.ones((32, 32, 10), dtype=bool)  # Wrong shape
        with pytest.raises(DataValidationError, match="must match spatial"):
            PerfusionDataset(
                data=sample_3d_data,
                affine=sample_affine,
                modality=Modality.DCE,
                quality_mask=bad_mask,
            )

    def test_acquisition_params(
        self, sample_4d_data: np.ndarray, sample_affine: np.ndarray
    ) -> None:
        """Test acquisition params handling."""
        time_points = np.linspace(0, 300, 30)
        params = DCEAcquisitionParams(
            tr=5.0,
            te=2.0,
            flip_angles=[2, 5, 10, 15, 20],
        )
        dataset = PerfusionDataset(
            data=sample_4d_data,
            affine=sample_affine,
            modality=Modality.DCE,
            time_points=time_points,
            acquisition_params=params,
        )
        assert dataset.acquisition_params.tr == 5.0
        assert isinstance(dataset.acquisition_params, DCEAcquisitionParams)
