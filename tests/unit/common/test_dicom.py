"""Unit tests for osipy.common.io.dicom module.

Tests cover:
- Error handling for missing paths and empty directories
- Real DICOM loading from ``data/test_dicom/`` (skipped when data absent)
"""

from pathlib import Path

import numpy as np
import pytest

from osipy.common.exceptions import IOError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_TEST_DICOM_DIR = DATA_DIR / "test_dicom"


def _skip_unless(path: Path) -> None:
    """``pytest.skip`` when *path* does not exist."""
    if not path.exists():
        pytest.skip(f"Data not found: {path}")


# ---------------------------------------------------------------------------
# Error handling (no real data needed)
# ---------------------------------------------------------------------------


class TestLoadDicom:
    """Tests for load_dicom function."""

    def test_path_not_found(self) -> None:
        """Test that missing path raises FileNotFoundError."""
        from osipy.common.io.dicom import load_dicom

        with pytest.raises(FileNotFoundError):
            load_dicom("/nonexistent/path")

    def test_no_dicom_files(self, tmp_path: Path) -> None:
        """Test that directory with no DICOM files raises IOError."""
        from osipy.common.io.dicom import load_dicom

        # Create empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(IOError, match="No valid DICOM"):
            load_dicom(empty_dir)


# ---------------------------------------------------------------------------
# Real DICOM data tests (skip when data/test_dicom/ is absent)
# ---------------------------------------------------------------------------


@pytest.mark.localdata
class TestLoadDicomRealData:
    """Load real DICOM files from ``data/test_dicom/``."""

    @pytest.mark.parametrize(
        "vendor",
        ["ge", "siemens", "philips"],
    )
    def test_load_vendor_dce(self, vendor: str) -> None:
        """load_dicom returns a PerfusionDataset for each vendor's DCE data."""
        from osipy.common.dataset import PerfusionDataset
        from osipy.common.io.dicom import load_dicom

        vendor_dir = _TEST_DICOM_DIR / vendor / "dce"
        _skip_unless(vendor_dir)

        # Walk down to find the first directory that actually contains .dcm files
        dcm_dir = _find_dcm_leaf(vendor_dir)
        if dcm_dir is None:
            pytest.skip(f"No .dcm files found under {vendor_dir}")

        ds = load_dicom(dcm_dir, prompt_missing=False)
        assert isinstance(ds, PerfusionDataset)
        assert ds.data.ndim >= 3, f"Expected >=3-D data, got {ds.data.ndim}-D"
        assert ds.data.size > 0
        assert np.isfinite(ds.data).any()

    @pytest.mark.parametrize(
        "vendor",
        ["ge", "siemens", "philips"],
    )
    def test_metadata_extracted(self, vendor: str) -> None:
        """Loaded dataset has non-empty acquisition metadata."""
        from osipy.common.io.dicom import load_dicom

        vendor_dir = _TEST_DICOM_DIR / vendor / "dce"
        _skip_unless(vendor_dir)

        dcm_dir = _find_dcm_leaf(vendor_dir)
        if dcm_dir is None:
            pytest.skip(f"No .dcm files found under {vendor_dir}")

        ds = load_dicom(dcm_dir, prompt_missing=False)
        acq = ds.acquisition_params
        assert acq is not None, "acquisition_params should not be None"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_dcm_leaf(root: Path) -> Path | None:
    """Return the first directory under *root* that contains ``.dcm`` files."""
    for p in sorted(root.rglob("*.dcm")):
        return p.parent
    return None
