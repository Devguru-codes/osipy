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


# ---------------------------------------------------------------------------
# _discover_dce_dicom_flat — flat-directory (PACS/vendor-export) layouts
# ---------------------------------------------------------------------------


def _make_synthetic_dicom(
    path: Path,
    *,
    series_uid: str,
    series_description: str,
    flip_angle: float,
    repetition_time: float,
    temporal_position: int | None,
    slice_location: float,
) -> None:
    """Write a minimal valid DICOM to *path* for discovery tests.

    Only the tags the flat discovery reads are populated; pixel data is
    a 1×1 uint16 zero (not used during header scans). ``series_uid``
    should be a valid DICOM UID (dot-separated digits only).
    """
    import warnings

    import numpy as np
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SeriesDescription = series_description
    ds.FlipAngle = flip_angle
    ds.RepetitionTime = repetition_time
    ds.EchoTime = 4.0
    ds.SliceLocation = slice_location
    if temporal_position is not None:
        ds.TemporalPositionIdentifier = temporal_position
    ds.Rows = 1
    ds.Columns = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.zeros((1, 1), dtype=np.uint16).tobytes()

    # Silence the noisy pydicom deprecation warnings that pytest promotes
    # to errors under this project's warning-filter configuration.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.save_as(str(path), write_like_original=False)


class TestDiscoverDceDicomFlat:
    """Flat-directory DCE discovery groups by SeriesInstanceUID and
    classifies dynamic vs VFA from headers alone — no subdirectory
    naming convention required."""

    def test_classifies_dynamic_by_temporal_position_identifier(
        self, tmp_path: Path
    ) -> None:
        from osipy.cli.runner import _discover_dce_dicom_flat

        # Three VFA series (2 slices each, no TPI) — different flip angles
        # but same TR.
        for fa in (5.0, 10.0, 20.0):
            uid = f"1.2.3.100{int(fa)}"
            for i, loc in enumerate((0.0, 5.0)):
                _make_synthetic_dicom(
                    tmp_path / f"vfa_{int(fa)}_{i}.dcm",
                    series_uid=uid,
                    series_description=f"FA{int(fa)}",
                    flip_angle=fa,
                    repetition_time=20.0,
                    temporal_position=None,
                    slice_location=loc,
                )

        # Dynamic series: 2 slices × 3 TPIs = 6 files
        dyn_uid = "1.2.3.999"
        for tpi in (1, 2, 3):
            for i, loc in enumerate((0.0, 5.0)):
                _make_synthetic_dicom(
                    tmp_path / f"dyn_{tpi}_{i}.dcm",
                    series_uid=dyn_uid,
                    series_description="DCE",
                    flip_angle=30.0,
                    repetition_time=6.0,
                    temporal_position=tpi,
                    slice_location=loc,
                )

        result = _discover_dce_dicom_flat(tmp_path)

        assert result is not None
        vfa_lists, perfusion_files = result
        # VFA sorted ascending by flip angle
        assert len(vfa_lists) == 3
        assert [len(v) for v in vfa_lists] == [2, 2, 2]
        # Dynamic carries 6 files, identified via TemporalPositionIdentifier
        assert len(perfusion_files) == 6

    def test_returns_none_for_single_series(self, tmp_path: Path) -> None:
        from osipy.cli.runner import _discover_dce_dicom_flat

        for i in range(3):
            _make_synthetic_dicom(
                tmp_path / f"f{i}.dcm",
                series_uid="1.2.3.500",
                series_description="FA10",
                flip_angle=10.0,
                repetition_time=20.0,
                temporal_position=None,
                slice_location=float(i),
            )

        assert _discover_dce_dicom_flat(tmp_path) is None

    def test_returns_none_when_no_vfa_candidates(self, tmp_path: Path) -> None:
        """Only a dynamic series + one other series without FlipAngle —
        nothing qualifies as VFA, so discovery returns None."""
        from osipy.cli.runner import _discover_dce_dicom_flat

        # Dynamic
        for tpi in (1, 2):
            _make_synthetic_dicom(
                tmp_path / f"dyn_{tpi}.dcm",
                series_uid="1.2.3.888",
                series_description="DCE",
                flip_angle=30.0,
                repetition_time=6.0,
                temporal_position=tpi,
                slice_location=0.0,
            )
        # A second series also with multiple TPIs → classifier will pick
        # one as dynamic; the other has no FA-only distinguishing.
        # We deliberately omit FlipAngle on this second one.
        import numpy as np
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(
            str(tmp_path / "other.dcm"),
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )
        ds.SeriesInstanceUID = "1.2.3.777"
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Rows = 1
        ds.Columns = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = np.zeros((1, 1), dtype=np.uint16).tobytes()
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            ds.save_as(str(tmp_path / "other.dcm"), write_like_original=False)

        assert _discover_dce_dicom_flat(tmp_path) is None

    def test_nested_discovery_falls_through_to_flat(self, tmp_path: Path) -> None:
        """The public _discover_dce_dicom_series delegates to flat
        discovery when no TCIA-style subdirs match."""
        from osipy.cli.runner import _discover_dce_dicom_series

        # Build the same flat layout as the classifier test
        for fa in (5.0, 10.0):
            uid = f"1.2.3.100{int(fa)}"
            _make_synthetic_dicom(
                tmp_path / f"vfa_{int(fa)}.dcm",
                series_uid=uid,
                series_description=f"FA{int(fa)}",
                flip_angle=fa,
                repetition_time=20.0,
                temporal_position=None,
                slice_location=0.0,
            )
        for tpi in (1, 2):
            _make_synthetic_dicom(
                tmp_path / f"dyn_{tpi}.dcm",
                series_uid="1.2.3.666",
                series_description="DCE",
                flip_angle=30.0,
                repetition_time=6.0,
                temporal_position=tpi,
                slice_location=0.0,
            )

        result = _discover_dce_dicom_series(tmp_path)
        assert result is not None
        vfa_lists, perfusion_files = result
        assert len(vfa_lists) == 2
        assert len(perfusion_files) == 2
