"""Vendor auto-detection for DICOM files.

This module provides automatic detection of scanner vendor
from DICOM metadata and returns the appropriate parser.

"""

import logging
from typing import Any

from osipy.common.io.vendors.base import VendorMetadata, VendorParser
from osipy.common.io.vendors.ge import GEParser
from osipy.common.io.vendors.philips import PhilipsParser
from osipy.common.io.vendors.siemens import SiemensParser

logger = logging.getLogger(__name__)


# Singleton parser instances
_PARSERS: list[VendorParser] = [
    SiemensParser(),
    GEParser(),
    PhilipsParser(),
]


def detect_vendor(dcm: Any) -> str:
    """Detect scanner vendor from DICOM dataset.

    Parameters
    ----------
    dcm : pydicom.Dataset
        DICOM dataset with Manufacturer tag.

    Returns
    -------
    str
        Vendor name: 'Siemens', 'GE', 'Philips', or 'Unknown'.

    Examples
    --------
    >>> import pydicom
    >>> dcm = pydicom.dcmread("scan.dcm")
    >>> vendor = detect_vendor(dcm)
    >>> print(vendor)
    'Siemens'
    """
    if not hasattr(dcm, "Manufacturer"):
        return "Unknown"

    manufacturer = str(dcm.Manufacturer).upper()

    if "SIEMENS" in manufacturer:
        return "Siemens"
    elif "GE" in manufacturer or "GENERAL ELECTRIC" in manufacturer:
        return "GE"
    elif "PHILIPS" in manufacturer:
        return "Philips"
    else:
        return "Unknown"


def get_vendor_parser(dcm: Any) -> VendorParser | None:
    """Get the appropriate vendor parser for a DICOM dataset.

    Parameters
    ----------
    dcm : pydicom.Dataset
        DICOM dataset to parse.

    Returns
    -------
    VendorParser | None
        Matching vendor parser, or None if vendor not supported.

    Examples
    --------
    >>> import pydicom
    >>> dcm = pydicom.dcmread("siemens_scan.dcm")
    >>> parser = get_vendor_parser(dcm)
    >>> if parser:
    ...     metadata = parser.extract_metadata(dcm)
    """
    for parser in _PARSERS:
        if parser.can_parse(dcm):
            return parser
    return None


def extract_vendor_metadata(dcm: Any) -> VendorMetadata:
    """Extract vendor-specific metadata from DICOM dataset.

    This is a convenience function that auto-detects the vendor
    and extracts metadata using the appropriate parser.

    Parameters
    ----------
    dcm : pydicom.Dataset
        DICOM dataset to parse.

    Returns
    -------
    VendorMetadata
        Extracted metadata. If vendor is not recognized,
        returns metadata with only standard DICOM fields.

    Examples
    --------
    >>> import pydicom
    >>> dcm = pydicom.dcmread("scan.dcm")
    >>> metadata = extract_vendor_metadata(dcm)
    >>> print(f"Vendor: {metadata.vendor}")
    >>> print(f"TR: {metadata.tr} ms")
    """
    parser = get_vendor_parser(dcm)

    if parser is not None:
        return parser.extract_metadata(dcm)

    # Fallback: extract standard DICOM tags only
    logger.warning("Unknown vendor, extracting standard DICOM tags only")
    return _extract_standard_only(dcm)


def _extract_standard_only(dcm: Any) -> VendorMetadata:
    """Extract only standard DICOM tags (fallback for unknown vendors).

    Parameters
    ----------
    dcm : pydicom.Dataset
        DICOM dataset.

    Returns
    -------
    VendorMetadata
        Metadata with standard DICOM fields only.
    """
    metadata = VendorMetadata(vendor="Unknown")

    # Software version
    if hasattr(dcm, "SoftwareVersions"):
        versions = dcm.SoftwareVersions
        if isinstance(versions, (list, tuple)):
            metadata.software_version = str(versions[0])
        else:
            metadata.software_version = str(versions)

    # Manufacturer (even if unknown category)
    if hasattr(dcm, "Manufacturer"):
        metadata.vendor = str(dcm.Manufacturer)

    # Basic timing parameters
    if hasattr(dcm, "RepetitionTime"):
        metadata.tr = float(dcm.RepetitionTime)
    if hasattr(dcm, "EchoTime"):
        metadata.te = float(dcm.EchoTime)
    if hasattr(dcm, "InversionTime"):
        metadata.ti = float(dcm.InversionTime)
    if hasattr(dcm, "FlipAngle"):
        metadata.flip_angle = float(dcm.FlipAngle)
    if hasattr(dcm, "MagneticFieldStrength"):
        metadata.field_strength = float(dcm.MagneticFieldStrength)

    # Sequence information
    if hasattr(dcm, "SequenceName"):
        metadata.sequence_name = str(dcm.SequenceName)
    if hasattr(dcm, "ScanningSequence"):
        metadata.sequence_type = str(dcm.ScanningSequence)

    # Standard diffusion tags
    if hasattr(dcm, "DiffusionBValue"):
        import numpy as np

        metadata.b_values = np.array([float(dcm.DiffusionBValue)])
    if hasattr(dcm, "DiffusionGradientOrientation"):
        import numpy as np

        try:
            vec = dcm.DiffusionGradientOrientation
            if len(vec) >= 3:
                metadata.b_vectors = np.array([[float(vec[i]) for i in range(3)]])
        except (ValueError, TypeError, IndexError):
            pass

    return metadata


def list_supported_vendors() -> list[str]:
    """List all supported scanner vendors.

    Returns
    -------
    list[str]
        List of supported vendor names.
    """
    return [parser.VENDOR_NAME for parser in _PARSERS]
