"""Vendor-specific DICOM parsing for osipy.

This subpackage provides vendor-specific metadata extraction for
DICOM data from Siemens, GE, and Philips scanners.

References
----------
DICOM Standard: https://www.dicomstandard.org/
"""

from osipy.common.io.vendors.base import VendorMetadata, VendorParser
from osipy.common.io.vendors.detection import detect_vendor, get_vendor_parser
from osipy.common.io.vendors.ge import GEParser
from osipy.common.io.vendors.philips import PhilipsParser
from osipy.common.io.vendors.siemens import SiemensParser

__all__ = [
    "GEParser",
    "PhilipsParser",
    "SiemensParser",
    "VendorMetadata",
    "VendorParser",
    "detect_vendor",
    "get_vendor_parser",
]
