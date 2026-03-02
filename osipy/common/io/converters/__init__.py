"""Data format converters for osipy.

This subpackage provides converters between different imaging formats,
including dcm2niix integration for DICOM to NIfTI conversion.
"""

from osipy.common.io.converters.dcm2niix import Dcm2niixConverter

__all__ = ["Dcm2niixConverter"]
