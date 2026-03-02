"""Siemens-specific DICOM parsing for osipy.

This module extracts metadata from Siemens MRI scanners,
including CSA header parsing for private tags.

References
----------
Siemens DICOM Conformance Statement
Private Tags: Groups 0019, 0021, 0029, 0051
"""

import contextlib
import logging
import re
from typing import Any

import numpy as np

from osipy.common.io.vendors.base import VendorMetadata, VendorParser

logger = logging.getLogger(__name__)


class SiemensParser(VendorParser):
    """Parser for Siemens MRI DICOM files.

    Extracts metadata from Siemens-specific private tags including
    CSA headers, sequence parameters, and diffusion information.

    Notes
    -----
    Siemens stores extended metadata in CSA headers within private
    tags. Common private groups:
    - 0019: Sequence parameters, diffusion info
    - 0021: Series and acquisition parameters
    - 0029: CSA headers (Series and Image)
    - 0051: Scanner-specific acquisition info

    Examples
    --------
    >>> from osipy.common.io.vendors.siemens import SiemensParser
    >>> parser = SiemensParser()
    >>> if parser.can_parse(dcm):
    ...     metadata = parser.extract_metadata(dcm)
    """

    VENDOR_NAME = "Siemens"

    # Private tag locations
    _CSA_SERIES_HEADER = (0x0029, 0x1010)  # CSA Series Header Info
    _CSA_IMAGE_HEADER = (0x0029, 0x1020)  # CSA Image Header Info
    _B_VALUE_TAG = (0x0019, 0x100C)  # Diffusion b-value
    _B_VECTOR_TAG = (0x0019, 0x100E)  # Diffusion gradient direction
    _GRADIENT_MODE = (0x0019, 0x100F)  # Diffusion gradient mode

    def can_parse(self, dcm: Any) -> bool:
        """Check if DICOM is from a Siemens scanner.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset to check.

        Returns
        -------
        bool
            True if Manufacturer tag contains 'SIEMENS'.
        """
        if not hasattr(dcm, "Manufacturer"):
            return False
        return "SIEMENS" in str(dcm.Manufacturer).upper()

    def extract_metadata(self, dcm: Any) -> VendorMetadata:
        """Extract Siemens-specific metadata from DICOM.

        Parameters
        ----------
        dcm : pydicom.Dataset
            Siemens DICOM dataset.

        Returns
        -------
        VendorMetadata
            Extracted metadata including CSA header info.
        """
        # Start with standard DICOM tags
        metadata = self._get_standard_metadata(dcm)

        # Extract diffusion parameters
        self._extract_diffusion_params(dcm, metadata)

        # Extract ASL parameters from CSA headers
        self._extract_asl_params(dcm, metadata)

        # Extract timing parameters
        self._extract_timing_params(dcm, metadata)

        # Extract sequence-specific info
        self._extract_sequence_info(dcm, metadata)

        return metadata

    def _extract_diffusion_params(self, dcm: Any, metadata: VendorMetadata) -> None:
        """Extract diffusion/IVIM parameters from Siemens private tags.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset.
        metadata : VendorMetadata
            Metadata object to populate.
        """
        # B-value from private tag (0019, 100C)
        b_value = self._safe_get_private_tag(dcm, 0x0019, 0x100C)
        if b_value is not None:
            with contextlib.suppress(ValueError, TypeError):
                metadata.b_values = np.array([float(b_value)])

        # B-vector from private tag (0019, 100E)
        b_vector = self._safe_get_private_tag(dcm, 0x0019, 0x100E)
        if b_vector is not None:
            try:
                if hasattr(b_vector, "__len__") and len(b_vector) >= 3:
                    metadata.b_vectors = np.array(
                        [[float(b_vector[i]) for i in range(3)]]
                    )
            except (ValueError, TypeError, IndexError):
                pass

        # Also check standard DICOM diffusion tags
        if metadata.b_values is None and hasattr(dcm, "DiffusionBValue"):
            metadata.b_values = np.array([float(dcm.DiffusionBValue)])

    def _extract_asl_params(self, dcm: Any, metadata: VendorMetadata) -> None:
        """Extract ASL parameters from Siemens DICOM.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset.
        metadata : VendorMetadata
            Metadata object to populate.
        """
        # Check sequence name for ASL indicators
        seq_name = metadata.sequence_name or ""
        seq_name_lower = seq_name.lower()

        # Detect ASL labeling type from sequence name
        if "pcasl" in seq_name_lower or "pseudo" in seq_name_lower:
            metadata.labeling_type = "PCASL"
        elif "pasl" in seq_name_lower or "fair" in seq_name_lower:
            metadata.labeling_type = "PASL"
        elif "vs" in seq_name_lower and "asl" in seq_name_lower:
            metadata.labeling_type = "VSASL"
        elif "asl" in seq_name_lower:
            # Generic ASL detection
            metadata.labeling_type = "PCASL"  # Default assumption

        # Try to extract from CSA headers
        csa_data = self._parse_csa_header(dcm)
        if csa_data:
            # Post-labeling delay
            if "PostLabelDelay" in csa_data:
                with contextlib.suppress(ValueError, TypeError):
                    metadata.post_labeling_delay = float(csa_data["PostLabelDelay"])

            # Labeling duration
            if "LabelingDuration" in csa_data:
                with contextlib.suppress(ValueError, TypeError):
                    metadata.labeling_duration = float(csa_data["LabelingDuration"])

            # Background suppression
            if "BackgroundSuppressionUsed" in csa_data:
                metadata.background_suppression = csa_data[
                    "BackgroundSuppressionUsed"
                ] in [
                    "YES",
                    "1",
                    True,
                ]

            # Bolus cutoff (QUIPSS)
            if "BolusCutOff" in csa_data:
                metadata.bolus_cutoff_flag = csa_data["BolusCutOff"] in [
                    "YES",
                    "1",
                    True,
                ]

            if "BolusCutOffDelay" in csa_data:
                with contextlib.suppress(ValueError, TypeError):
                    metadata.bolus_cutoff_delay = float(csa_data["BolusCutOffDelay"])

        # Extract inversion time as potential PLD for PASL
        if (
            metadata.post_labeling_delay is None
            and metadata.ti is not None
            and metadata.labeling_type in ["PASL", "FAIR"]
        ):
            metadata.post_labeling_delay = metadata.ti

    def _extract_timing_params(self, dcm: Any, metadata: VendorMetadata) -> None:
        """Extract timing parameters from Siemens DICOM.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset.
        metadata : VendorMetadata
            Metadata object to populate.
        """
        # Try to get slice timing from CSA
        csa_data = self._parse_csa_header(dcm)
        if csa_data and "SliceTiming" in csa_data:
            timing = csa_data["SliceTiming"]
            if isinstance(timing, (list, tuple)):
                metadata.slice_timing = [float(t) for t in timing]

        # Check for MosaicRefAcqTimes in private tags
        mosaic_times = self._safe_get_private_tag(dcm, 0x0019, 0x1029)
        if mosaic_times is not None:
            try:
                if hasattr(mosaic_times, "__iter__"):
                    metadata.slice_timing = [float(t) for t in mosaic_times]
            except (ValueError, TypeError):
                pass

    def _extract_sequence_info(self, dcm: Any, metadata: VendorMetadata) -> None:
        """Extract sequence-specific information.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset.
        metadata : VendorMetadata
            Metadata object to populate.
        """
        # Protocol name
        if hasattr(dcm, "ProtocolName"):
            metadata.extra["protocol_name"] = str(dcm.ProtocolName)

        # Series description
        if hasattr(dcm, "SeriesDescription"):
            metadata.extra["series_description"] = str(dcm.SeriesDescription)

        # Siemens-specific sequence type from private tag
        seq_variant = self._safe_get_private_tag(dcm, 0x0019, 0x109C)
        if seq_variant is not None:
            metadata.extra["sequence_variant"] = str(seq_variant)

        # Parallel imaging info
        pat_mode = self._safe_get_private_tag(dcm, 0x0051, 0x1011)
        if pat_mode is not None:
            metadata.extra["parallel_imaging_mode"] = str(pat_mode)

    def _parse_csa_header(self, dcm: Any) -> dict[str, Any]:
        """Parse Siemens CSA header for extended parameters.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset.

        Returns
        -------
        dict[str, Any]
            Parsed CSA header parameters.
        """
        result: dict[str, Any] = {}

        # Try series header first, then image header
        for tag in [self._CSA_SERIES_HEADER, self._CSA_IMAGE_HEADER]:
            if tag in dcm:
                try:
                    csa_data = dcm[tag].value
                    if csa_data:
                        parsed = self._decode_csa_header(csa_data)
                        result.update(parsed)
                except (AttributeError, ValueError) as e:
                    logger.debug(f"Failed to parse CSA header {tag}: {e}")

        return result

    def _decode_csa_header(self, csa_bytes: bytes) -> dict[str, Any]:
        """Decode Siemens CSA header binary data.

        Parameters
        ----------
        csa_bytes : bytes
            Raw CSA header bytes.

        Returns
        -------
        dict[str, Any]
            Decoded parameter dictionary.

        Notes
        -----
        CSA headers have a specific binary format. This is a simplified
        parser that extracts text-based parameters using regex patterns.
        """
        result: dict[str, Any] = {}

        if not csa_bytes:
            return result

        try:
            # Convert to string, handling encoding issues
            if isinstance(csa_bytes, bytes):
                text = csa_bytes.decode("latin-1", errors="ignore")
            else:
                text = str(csa_bytes)

            # Look for common ASL-related parameters
            patterns = {
                "PostLabelDelay": r"PostLabelDelay[^\d]*(\d+\.?\d*)",
                "LabelingDuration": r"LabelingDuration[^\d]*(\d+\.?\d*)",
                "BackgroundSuppressionUsed": r"BackgroundSuppression[^\w]*(YES|NO|1|0)",
                "BolusCutOff": r"BolusCutOff[^\w]*(YES|NO|1|0)",
                "BolusCutOffDelay": r"BolusCutOffDelay[^\d]*(\d+\.?\d*)",
            }

            for param_name, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result[param_name] = match.group(1)

        except (UnicodeDecodeError, AttributeError) as e:
            logger.debug(f"CSA header decode error: {e}")

        return result
