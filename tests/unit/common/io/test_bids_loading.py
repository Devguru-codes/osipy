"""Tests for BIDS input loading.

Tests the load_bids, load_bids_with_m0, and related functions.
"""

import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from osipy.common.io.bids import (
    is_bids_dataset,
    load_asl_context,
    load_bids,
    load_bids_with_m0,
)
from osipy.common.types import Modality


@pytest.fixture
def temp_bids_dataset():
    """Create a temporary BIDS dataset structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bids_root = Path(tmpdir)

        # Create dataset_description.json
        desc = {
            "Name": "Test Dataset",
            "BIDSVersion": "1.9.0",
            "DatasetType": "raw",
        }
        with (bids_root / "dataset_description.json").open("w") as f:
            json.dump(desc, f)

        # Create subject directory structure
        perf_dir = bids_root / "sub-01" / "perf"
        perf_dir.mkdir(parents=True)

        # Create test NIfTI file
        data = np.random.randn(64, 64, 24, 80).astype(np.float32)
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        nib.save(nifti_img, perf_dir / "sub-01_asl.nii.gz")

        # Create sidecar JSON
        sidecar = {
            "RepetitionTimePreparation": 4.5,
            "ArterialSpinLabelingType": "PCASL",
            "PostLabelingDelay": 1.8,
            "LabelingDuration": 1.8,
            "BackgroundSuppression": True,
            "MagneticFieldStrength": 3.0,
        }
        with (perf_dir / "sub-01_asl.json").open("w") as f:
            json.dump(sidecar, f)

        # Create aslcontext.tsv (proper format: header row then data rows)
        context_lines = ["volume_type"]  # Header
        for _ in range(40):
            context_lines.append("control")
            context_lines.append("label")
        with (perf_dir / "sub-01_aslcontext.tsv").open("w") as f:
            f.write("\n".join(context_lines))

        yield bids_root


@pytest.fixture
def temp_bids_with_m0(temp_bids_dataset):
    """Extend BIDS dataset with M0 scan."""
    bids_root = temp_bids_dataset
    perf_dir = bids_root / "sub-01" / "perf"

    # Create M0 NIfTI
    m0_data = np.random.randn(64, 64, 24).astype(np.float32)
    affine = np.eye(4)
    m0_img = nib.Nifti1Image(m0_data, affine)
    nib.save(m0_img, perf_dir / "sub-01_m0scan.nii.gz")

    # Create M0 sidecar
    m0_sidecar = {
        "IntendedFor": "sub-01/perf/sub-01_asl.nii.gz",
    }
    with (perf_dir / "sub-01_m0scan.json").open("w") as f:
        json.dump(m0_sidecar, f)

    return bids_root


class TestIsBidsDataset:
    """Tests for is_bids_dataset function."""

    def test_valid_bids_dataset(self, temp_bids_dataset):
        """Test detection of valid BIDS dataset."""
        assert is_bids_dataset(temp_bids_dataset)

    def test_non_bids_directory(self, tmp_path):
        """Test that non-BIDS directory returns False."""
        assert not is_bids_dataset(tmp_path)

    def test_file_path(self, temp_bids_dataset):
        """Test that file path returns False."""
        file_path = temp_bids_dataset / "dataset_description.json"
        assert not is_bids_dataset(file_path)


class TestLoadBids:
    """Tests for load_bids function."""

    def test_load_asl(self, temp_bids_dataset):
        """Test loading ASL data from BIDS."""
        dataset = load_bids(
            temp_bids_dataset,
            subject="01",
            modality=Modality.ASL,
            interactive=False,
        )

        assert dataset.shape == (64, 64, 24, 80)
        assert dataset.modality == Modality.ASL
        assert dataset.source_format == "bids"

    def test_load_with_params(self, temp_bids_dataset):
        """Test that acquisition params are extracted."""
        dataset = load_bids(
            temp_bids_dataset,
            subject="01",
            modality=Modality.ASL,
            interactive=False,
        )

        # Check acquisition params from sidecar
        assert dataset.acquisition_params is not None
        assert dataset.acquisition_params.pld == 1800.0  # From sidecar (1.8s * 1000)

    def test_load_nonexistent_subject(self, temp_bids_dataset):
        """Test error for non-existent subject."""
        with pytest.raises(FileNotFoundError):
            load_bids(
                temp_bids_dataset,
                subject="99",
                modality=Modality.ASL,
            )

    def test_load_nonexistent_directory(self):
        """Test error for non-existent directory."""
        with pytest.raises(FileNotFoundError):
            load_bids(
                "/nonexistent/path",
                subject="01",
                modality=Modality.ASL,
            )


class TestLoadBidsWithM0:
    """Tests for load_bids_with_m0 function."""

    def test_load_with_m0(self, temp_bids_with_m0):
        """Test loading ASL data with M0."""
        asl_data, m0_data = load_bids_with_m0(
            temp_bids_with_m0,
            subject="01",
            interactive=False,
        )

        assert asl_data.shape == (64, 64, 24, 80)
        assert m0_data is not None
        assert m0_data.shape == (64, 64, 24)

    def test_load_without_m0(self, temp_bids_dataset):
        """Test loading when M0 not present."""
        asl_data, m0_data = load_bids_with_m0(
            temp_bids_dataset,
            subject="01",
            interactive=False,
        )

        assert asl_data.shape == (64, 64, 24, 80)
        assert m0_data is None


class TestLoadAslContext:
    """Tests for load_asl_context function."""

    def test_load_context(self, temp_bids_dataset):
        """Test loading ASL context."""
        context = load_asl_context(
            temp_bids_dataset,
            subject="01",
        )

        assert len(context) == 80
        assert "control" in context
        assert "label" in context

    def test_context_not_found(self, tmp_path):
        """Test error when context file not found."""
        # Create minimal BIDS structure without context file
        (tmp_path / "sub-01" / "perf").mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            load_asl_context(tmp_path, subject="01")
