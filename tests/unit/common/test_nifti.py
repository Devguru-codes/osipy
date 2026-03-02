"""Unit tests for osipy.common.io.nifti module."""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from osipy.common.exceptions import IOError
from osipy.common.io.nifti import load_nifti
from osipy.common.types import Modality


class TestLoadNifti:
    """Tests for load_nifti function."""

    @pytest.fixture
    def temp_nifti_3d(self) -> Path:
        """Create temporary 3D NIfTI file."""
        data = np.random.rand(64, 64, 20).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            nib.save(img, f.name)
            return Path(f.name)

    @pytest.fixture
    def temp_nifti_4d(self) -> Path:
        """Create temporary 4D NIfTI file."""
        data = np.random.rand(64, 64, 20, 30).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        # Set TR in header for time calculation
        img.header.set_zooms((2.0, 2.0, 3.0, 5.0))

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            nib.save(img, f.name)
            return Path(f.name)

    def test_load_3d_nifti(self, temp_nifti_3d: Path) -> None:
        """Test loading 3D NIfTI file."""
        dataset = load_nifti(temp_nifti_3d, modality=Modality.DCE)

        assert dataset.shape == (64, 64, 20)
        assert dataset.is_dynamic is False
        assert dataset.modality == Modality.DCE
        assert dataset.source_format == "nifti"

    def test_load_4d_nifti(self, temp_nifti_4d: Path) -> None:
        """Test loading 4D NIfTI file."""
        dataset = load_nifti(temp_nifti_4d, modality=Modality.DCE)

        assert dataset.shape == (64, 64, 20, 30)
        assert dataset.is_dynamic is True
        assert dataset.n_timepoints == 30
        assert dataset.time_points is not None
        assert len(dataset.time_points) == 30

    def test_file_not_found(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_nifti("/nonexistent/path.nii.gz")

    def test_invalid_extension(self, tmp_path: Path) -> None:
        """Test that invalid extension raises IOError."""
        bad_file = tmp_path / "test.txt"
        bad_file.write_text("not a nifti file")
        with pytest.raises(IOError, match="Invalid NIfTI"):
            load_nifti(bad_file)

    def test_affine_preserved(self, temp_nifti_3d: Path) -> None:
        """Test that affine is correctly loaded."""
        dataset = load_nifti(temp_nifti_3d)
        assert dataset.affine.shape == (4, 4)
        np.testing.assert_array_equal(dataset.affine, np.eye(4))
