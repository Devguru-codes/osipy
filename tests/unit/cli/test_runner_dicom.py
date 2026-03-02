"""Tests for DICOM compatibility in the CLI pipeline runner.

Tests cover:
- ``_detect_multi_series_layout()`` detection of multi-series DICOM dirs
- ``_load_data()`` delegation to ``load_perfusion()`` with correct args
- ``_detect_format()`` on directories with nested DICOM subdirectories
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# _detect_multi_series_layout
# ---------------------------------------------------------------------------


class TestDetectMultiSeriesLayout:
    """Tests for osipy.common.io.load._detect_multi_series_layout."""

    def test_multi_series_returns_sorted_dirs(self, tmp_path: Path) -> None:
        """Directories with 2+ subdirs containing .dcm files are multi-series."""
        from osipy.common.io.load import _detect_multi_series_layout

        # Create two subdirectories each with a .dcm file
        series_a = tmp_path / "MR_series_001"
        series_b = tmp_path / "MR_series_002"
        series_a.mkdir()
        series_b.mkdir()
        (series_a / "001.dcm").write_bytes(b"fake")
        (series_b / "001.dcm").write_bytes(b"fake")

        result = _detect_multi_series_layout(tmp_path)
        assert result is not None
        assert len(result) == 2
        assert result == sorted(result)

    def test_single_series_returns_none(self, tmp_path: Path) -> None:
        """Directory with .dcm files directly in it is single-series."""
        from osipy.common.io.load import _detect_multi_series_layout

        (tmp_path / "001.dcm").write_bytes(b"fake")
        result = _detect_multi_series_layout(tmp_path)
        assert result is None

    def test_empty_dir_returns_none(self, tmp_path: Path) -> None:
        """Empty directory returns None."""
        from osipy.common.io.load import _detect_multi_series_layout

        result = _detect_multi_series_layout(tmp_path)
        assert result is None

    def test_single_subdir_returns_none(self, tmp_path: Path) -> None:
        """Only one subdir with DICOM is not multi-series (need 2+)."""
        from osipy.common.io.load import _detect_multi_series_layout

        series_a = tmp_path / "MR_series_001"
        series_a.mkdir()
        (series_a / "001.dcm").write_bytes(b"fake")

        result = _detect_multi_series_layout(tmp_path)
        assert result is None

    def test_file_path_returns_none(self, tmp_path: Path) -> None:
        """Non-directory path returns None."""
        from osipy.common.io.load import _detect_multi_series_layout

        f = tmp_path / "file.txt"
        f.write_text("data")

        result = _detect_multi_series_layout(f)
        assert result is None


# ---------------------------------------------------------------------------
# _load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    """Tests for osipy.cli.runner._load_data."""

    def test_passes_format_and_modality(self) -> None:
        """Verify format, modality, and interactive=False are forwarded."""
        from osipy.cli.config import DataConfig, PipelineConfig

        # _load_data does `from osipy.common.io.load import load_perfusion`
        # so we must patch the canonical location *before* calling _load_data.
        with patch("osipy.common.io.load.load_perfusion", create=True) as mock_lp:
            mock_lp.return_value = MagicMock()

            from osipy.cli.runner import _load_data

            config = PipelineConfig(
                modality="dce",
                data=DataConfig(format="dicom"),
            )

            _load_data(config, Path("/some/path"), "DCE")

            mock_lp.assert_called_once_with(
                path=Path("/some/path"),
                modality="DCE",
                format="dicom",
                subject=None,
                session=None,
                interactive=False,
            )

    @patch("osipy.common.io.load.load_perfusion")
    def test_passes_bids_fields(self, mock_lp: MagicMock) -> None:
        """Verify subject and session are forwarded for BIDS."""
        from osipy.cli.config import DataConfig, PipelineConfig
        from osipy.cli.runner import _load_data

        mock_lp.return_value = MagicMock()

        config = PipelineConfig(
            modality="dce",
            data=DataConfig(format="bids", subject="01", session="pre"),
        )

        _load_data(config, Path("/bids/root"), "DCE")

        mock_lp.assert_called_once_with(
            path=Path("/bids/root"),
            modality="DCE",
            format="bids",
            subject="01",
            session="pre",
            interactive=False,
        )

    @patch("osipy.common.io.load.load_perfusion")
    def test_auto_format_default(self, mock_lp: MagicMock) -> None:
        """Default format='auto' is passed through."""
        from osipy.cli.config import PipelineConfig
        from osipy.cli.runner import _load_data

        mock_lp.return_value = MagicMock()

        config = PipelineConfig(modality="dsc")

        _load_data(config, Path("/data.nii.gz"), "DSC")

        mock_lp.assert_called_once()
        assert mock_lp.call_args.kwargs["format"] == "auto"


# ---------------------------------------------------------------------------
# _detect_format with nested DICOM subdirectories
# ---------------------------------------------------------------------------


class TestDetectFormatMultiSeries:
    """Test _detect_format on multi-series DICOM directories."""

    def test_subdirs_with_dcm_detected_as_dicom(self, tmp_path: Path) -> None:
        """Directory with subdirectories containing .dcm → 'dicom'."""
        from osipy.common.io.load import _detect_format

        # Create study-level dir with series subdirs
        series_a = tmp_path / "MR_series_001"
        series_b = tmp_path / "MR_series_002"
        series_a.mkdir()
        series_b.mkdir()
        (series_a / "img001.dcm").write_bytes(b"fake")
        (series_b / "img001.dcm").write_bytes(b"fake")

        result = _detect_format(tmp_path)
        assert result == "dicom"


# ---------------------------------------------------------------------------
# _load_dicom multi-series delegation
# ---------------------------------------------------------------------------


class TestLoadDicomMultiSeries:
    """Test that _load_dicom delegates to load_dicom_multi_series."""

    def test_delegates_to_multi_series(self, tmp_path: Path) -> None:
        """When multi-series layout detected, delegate to load_dicom_multi_series."""
        from osipy.common.io.load import _load_dicom
        from osipy.common.types import Modality

        series_dirs = [tmp_path / "a", tmp_path / "b"]

        with (
            patch(
                "osipy.common.io.load._detect_multi_series_layout",
                return_value=series_dirs,
            ) as mock_detect,
            patch(
                "osipy.common.io.dicom.load_dicom_multi_series",
                return_value=MagicMock(),
            ) as mock_load_ms,
        ):
            _load_dicom(
                tmp_path,
                modality=Modality.DCE,
                interactive=False,
                use_dcm2niix=False,
            )

            mock_detect.assert_called_once_with(tmp_path)
            mock_load_ms.assert_called_once_with(
                series_dirs=series_dirs,
                prompt_missing=False,
                modality=Modality.DCE,
            )

    @patch("osipy.common.io.load._detect_multi_series_layout")
    def test_falls_through_when_no_multi_series(
        self, mock_detect: MagicMock, tmp_path: Path
    ) -> None:
        """When no multi-series layout, fall through to single-series path."""
        from osipy.common.exceptions import IOError as OsipyIOError
        from osipy.common.io.load import _load_dicom
        from osipy.common.types import Modality

        mock_detect.return_value = None

        # No DICOM files exist → should raise IOError from single-series path
        with pytest.raises((OsipyIOError, ImportError)):
            _load_dicom(
                tmp_path,
                modality=Modality.DCE,
                interactive=False,
                use_dcm2niix=False,
            )
