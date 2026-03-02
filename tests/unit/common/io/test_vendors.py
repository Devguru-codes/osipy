"""Tests for vendor-specific DICOM parsing.

Tests the vendor detection and metadata extraction for
Siemens, GE, and Philips scanners.
"""

from unittest.mock import MagicMock

import numpy as np

from osipy.common.io.vendors.base import VendorMetadata
from osipy.common.io.vendors.detection import (
    detect_vendor,
    extract_vendor_metadata,
    get_vendor_parser,
    list_supported_vendors,
)
from osipy.common.io.vendors.ge import GEParser
from osipy.common.io.vendors.philips import PhilipsParser
from osipy.common.io.vendors.siemens import SiemensParser


class TestVendorMetadata:
    """Tests for VendorMetadata dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        metadata = VendorMetadata()
        assert metadata.vendor == ""
        assert metadata.tr is None
        assert metadata.te is None
        assert metadata.labeling_type is None
        assert metadata.b_values is None

    def test_has_asl_params(self):
        """Test ASL parameter detection."""
        # No ASL params
        metadata = VendorMetadata()
        assert not metadata.has_asl_params()

        # With ASL params
        metadata = VendorMetadata(
            labeling_type="PCASL",
            post_labeling_delay=1800.0,
        )
        assert metadata.has_asl_params()

        # Partial params
        metadata = VendorMetadata(labeling_type="PCASL")
        assert not metadata.has_asl_params()

    def test_has_diffusion_params(self):
        """Test diffusion parameter detection."""
        # No diffusion params
        metadata = VendorMetadata()
        assert not metadata.has_diffusion_params()

        # With b-values
        metadata = VendorMetadata(b_values=np.array([0, 500, 1000]))
        assert metadata.has_diffusion_params()

        # Empty b-values
        metadata = VendorMetadata(b_values=np.array([]))
        assert not metadata.has_diffusion_params()

    def test_to_dict(self):
        """Test dictionary conversion."""
        metadata = VendorMetadata(
            vendor="Siemens",
            tr=5000.0,
            te=30.0,
            b_values=np.array([0, 500, 1000]),
        )
        result = metadata.to_dict()

        assert result["vendor"] == "Siemens"
        assert result["tr"] == 5000.0
        assert result["te"] == 30.0
        assert result["b_values"] == [0, 500, 1000]


class TestSiemensParser:
    """Tests for Siemens DICOM parser."""

    def test_can_parse_siemens(self):
        """Test detection of Siemens DICOM."""
        parser = SiemensParser()

        dcm = MagicMock()
        dcm.Manufacturer = "SIEMENS"
        assert parser.can_parse(dcm)

        dcm.Manufacturer = "Siemens Healthineers"
        assert parser.can_parse(dcm)

        dcm.Manufacturer = "GE MEDICAL SYSTEMS"
        assert not parser.can_parse(dcm)

    def test_can_parse_no_manufacturer(self):
        """Test handling of missing Manufacturer tag."""
        parser = SiemensParser()
        dcm = MagicMock(spec=[])  # No Manufacturer attribute
        assert not parser.can_parse(dcm)

    def test_extract_basic_metadata(self):
        """Test extraction of basic DICOM tags."""
        parser = SiemensParser()

        dcm = MagicMock()
        dcm.Manufacturer = "SIEMENS"
        dcm.RepetitionTime = 5000.0
        dcm.EchoTime = 30.0
        dcm.FlipAngle = 90.0
        dcm.MagneticFieldStrength = 3.0
        dcm.SequenceName = "ep2d_diff"

        # Mock private tag access
        dcm.__contains__ = lambda self, key: False

        metadata = parser.extract_metadata(dcm)

        assert metadata.vendor == "Siemens"
        assert metadata.tr == 5000.0
        assert metadata.te == 30.0
        assert metadata.flip_angle == 90.0
        assert metadata.field_strength == 3.0

    def test_extract_diffusion_b_value(self):
        """Test extraction of b-value from Siemens private tag."""
        parser = SiemensParser()

        dcm = MagicMock()
        dcm.Manufacturer = "SIEMENS"
        dcm.RepetitionTime = 3000.0
        dcm.EchoTime = 70.0

        # Mock private tag (0019, 100C) - b-value
        dcm.__contains__ = lambda self, key: key == (0x0019, 0x100C)
        mock_elem = MagicMock()
        mock_elem.value = 800.0
        dcm.__getitem__ = (
            lambda self, key: mock_elem if key == (0x0019, 0x100C) else None
        )

        metadata = parser.extract_metadata(dcm)

        assert metadata.b_values is not None
        assert metadata.b_values[0] == 800.0


class TestGEParser:
    """Tests for GE DICOM parser."""

    def test_can_parse_ge(self):
        """Test detection of GE DICOM."""
        parser = GEParser()

        dcm = MagicMock()
        dcm.Manufacturer = "GE MEDICAL SYSTEMS"
        assert parser.can_parse(dcm)

        dcm.Manufacturer = "General Electric"
        assert parser.can_parse(dcm)

        dcm.Manufacturer = "SIEMENS"
        assert not parser.can_parse(dcm)

    def test_extract_basic_metadata(self):
        """Test extraction of basic metadata."""
        parser = GEParser()

        dcm = MagicMock()
        dcm.Manufacturer = "GE MEDICAL SYSTEMS"
        dcm.RepetitionTime = 4500.0
        dcm.EchoTime = 12.0
        dcm.MagneticFieldStrength = 3.0
        dcm.SeriesDescription = "3D ASL"

        dcm.__contains__ = lambda self, key: False

        metadata = parser.extract_metadata(dcm)

        assert metadata.vendor == "GE"
        assert metadata.tr == 4500.0
        assert metadata.te == 12.0


class TestPhilipsParser:
    """Tests for Philips DICOM parser."""

    def test_can_parse_philips(self):
        """Test detection of Philips DICOM."""
        parser = PhilipsParser()

        dcm = MagicMock()
        dcm.Manufacturer = "Philips Medical Systems"
        assert parser.can_parse(dcm)

        dcm.Manufacturer = "PHILIPS"
        assert parser.can_parse(dcm)

        dcm.Manufacturer = "SIEMENS"
        assert not parser.can_parse(dcm)

    def test_extract_basic_metadata(self):
        """Test extraction of basic metadata."""
        parser = PhilipsParser()

        dcm = MagicMock()
        dcm.Manufacturer = "Philips"
        dcm.RepetitionTime = 5000.0
        dcm.EchoTime = 14.0
        dcm.MagneticFieldStrength = 3.0

        dcm.__contains__ = lambda self, key: False

        metadata = parser.extract_metadata(dcm)

        assert metadata.vendor == "Philips"
        assert metadata.tr == 5000.0
        assert metadata.te == 14.0


class TestVendorDetection:
    """Tests for vendor auto-detection."""

    def test_detect_siemens(self):
        """Test Siemens detection."""
        dcm = MagicMock()
        dcm.Manufacturer = "SIEMENS"
        assert detect_vendor(dcm) == "Siemens"

    def test_detect_ge(self):
        """Test GE detection."""
        dcm = MagicMock()
        dcm.Manufacturer = "GE MEDICAL SYSTEMS"
        assert detect_vendor(dcm) == "GE"

    def test_detect_philips(self):
        """Test Philips detection."""
        dcm = MagicMock()
        dcm.Manufacturer = "Philips"
        assert detect_vendor(dcm) == "Philips"

    def test_detect_unknown(self):
        """Test unknown vendor detection."""
        dcm = MagicMock()
        dcm.Manufacturer = "Unknown MRI Corp"
        assert detect_vendor(dcm) == "Unknown"

    def test_detect_no_manufacturer(self):
        """Test handling of missing Manufacturer."""
        dcm = MagicMock(spec=[])
        assert detect_vendor(dcm) == "Unknown"

    def test_get_vendor_parser_siemens(self):
        """Test getting Siemens parser."""
        dcm = MagicMock()
        dcm.Manufacturer = "SIEMENS"
        parser = get_vendor_parser(dcm)
        assert isinstance(parser, SiemensParser)

    def test_get_vendor_parser_unknown(self):
        """Test getting parser for unknown vendor."""
        dcm = MagicMock()
        dcm.Manufacturer = "Unknown"
        parser = get_vendor_parser(dcm)
        assert parser is None

    def test_list_supported_vendors(self):
        """Test listing supported vendors."""
        vendors = list_supported_vendors()
        assert "Siemens" in vendors
        assert "GE" in vendors
        assert "Philips" in vendors

    def test_extract_vendor_metadata_siemens(self):
        """Test metadata extraction with auto-detection."""
        dcm = MagicMock()
        dcm.Manufacturer = "SIEMENS"
        dcm.RepetitionTime = 5000.0
        dcm.EchoTime = 30.0
        dcm.__contains__ = lambda self, key: False

        metadata = extract_vendor_metadata(dcm)

        assert metadata.vendor == "Siemens"
        assert metadata.tr == 5000.0

    def test_extract_vendor_metadata_unknown(self):
        """Test metadata extraction for unknown vendor."""
        dcm = MagicMock()
        dcm.Manufacturer = "Unknown Corp"
        dcm.RepetitionTime = 5000.0
        dcm.EchoTime = 30.0

        metadata = extract_vendor_metadata(dcm)

        # Should still extract standard tags
        assert metadata.tr == 5000.0
        assert metadata.te == 30.0
