"""I/O utilities for osipy.

This module provides functions for loading and exporting
perfusion imaging data in various formats including NIfTI,
DICOM, and BIDS.

Key functions:
- `load_perfusion()`: Universal loader with auto-detection
- `load_nifti()`: Load NIfTI files
- `load_dicom()`: Load DICOM series
- `load_bids()`: Load from BIDS dataset
- `export_bids()`: Export to BIDS derivatives

"""

from osipy.common.io.bids import (
    export_bids,
    get_bids_subjects,
    is_bids_dataset,
    load_asl_context,
    load_bids,
    load_bids_with_m0,
)
from osipy.common.io.dicom import (
    build_affine_from_dicom,
    load_dicom,
    load_dicom_multi_series,
)
from osipy.common.io.load import load_perfusion
from osipy.common.io.nifti import load_nifti, save_nifti

__all__ = [
    "build_affine_from_dicom",
    "export_bids",
    "get_bids_subjects",
    "is_bids_dataset",
    "load_asl_context",
    # BIDS I/O
    "load_bids",
    "load_bids_with_m0",
    # DICOM I/O
    "load_dicom",
    "load_dicom_multi_series",
    # NIfTI I/O
    "load_nifti",
    # Universal loader
    "load_perfusion",
    "save_nifti",
]
