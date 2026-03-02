"""Base classes for vendor-specific DICOM parsing.

This module defines the abstract base class and common data structures
for extracting vendor-specific metadata from DICOM files.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class VendorMetadata:
    """Container for vendor-extracted DICOM metadata.

    This dataclass holds metadata extracted from vendor-specific
    private tags, providing a normalized interface for accessing
    acquisition parameters regardless of scanner manufacturer.

    Attributes
    ----------
    vendor : str
        Scanner manufacturer name.
    software_version : str | None
        Scanner software version.
    tr : float | None
        Repetition time in milliseconds.
    te : float | None
        Echo time in milliseconds.
    ti : float | None
        Inversion time in milliseconds (for IR sequences).
    flip_angle : float | None
        Flip angle in degrees.
    field_strength : float | None
        Magnetic field strength in Tesla.
    sequence_name : str | None
        Pulse sequence name.
    sequence_type : str | None
        Sequence type (e.g., 'GRE', 'SE', 'EPI').
    labeling_type : str | None
        ASL labeling type (e.g., 'PCASL', 'PASL', 'VSASL').
    post_labeling_delay : float | list[float] | None
        Post-labeling delay(s) in milliseconds for ASL.
    labeling_duration : float | None
        Labeling duration in milliseconds for ASL.
    b_values : NDArray[np.floating] | None
        Array of b-values in s/mm^2 for diffusion/IVIM.
    b_vectors : NDArray[np.floating] | None
        Array of b-vectors (Nx3) for diffusion.
    temporal_resolution : float | None
        Time between dynamic acquisitions in seconds.
    slice_timing : list[float] | None
        Slice acquisition times in milliseconds.
    bolus_cutoff_flag : bool | None
        Whether bolus cutoff (QUIPSS) was applied for PASL.
    bolus_cutoff_delay : float | None
        Bolus cutoff delay time in milliseconds.
    background_suppression : bool | None
        Whether background suppression was applied for ASL.
    bs_pulse_times : list[float] | None
        Background suppression pulse timing in milliseconds.
    venc : float | None
        Velocity encoding value in cm/s for velocity-selective ASL.
    m0_type : str | None
        M0 calibration type ('separate', 'included', 'absent').
    extra : dict[str, Any]
        Additional vendor-specific parameters not in standard fields.

    Notes
    -----
    All timing values are in milliseconds unless otherwise noted.
    Parameters that cannot be extracted are set to None.
    """

    vendor: str = ""
    software_version: str | None = None

    # Common acquisition parameters
    tr: float | None = None
    te: float | None = None
    ti: float | None = None
    flip_angle: float | None = None
    field_strength: float | None = None
    sequence_name: str | None = None
    sequence_type: str | None = None

    # ASL-specific parameters
    labeling_type: str | None = None
    post_labeling_delay: float | list[float] | None = None
    labeling_duration: float | None = None
    bolus_cutoff_flag: bool | None = None
    bolus_cutoff_delay: float | None = None
    background_suppression: bool | None = None
    bs_pulse_times: list[float] | None = None
    venc: float | None = None
    m0_type: str | None = None

    # Diffusion/IVIM parameters
    b_values: "NDArray[np.floating[Any]] | None" = None
    b_vectors: "NDArray[np.floating[Any]] | None" = None

    # Timing parameters
    temporal_resolution: float | None = None
    slice_timing: list[float] | None = None

    # Extra vendor-specific fields
    extra: dict[str, Any] = field(default_factory=dict)

    def has_asl_params(self) -> bool:
        """Check if ASL-specific parameters are available.

        Returns
        -------
        bool
            True if labeling_type and post_labeling_delay are set.
        """
        return self.labeling_type is not None and self.post_labeling_delay is not None

    def has_diffusion_params(self) -> bool:
        """Check if diffusion/IVIM parameters are available.

        Returns
        -------
        bool
            True if b_values are set.
        """
        return self.b_values is not None and len(self.b_values) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of metadata.
        """
        result = {
            "vendor": self.vendor,
            "software_version": self.software_version,
            "tr": self.tr,
            "te": self.te,
            "ti": self.ti,
            "flip_angle": self.flip_angle,
            "field_strength": self.field_strength,
            "sequence_name": self.sequence_name,
            "sequence_type": self.sequence_type,
            "labeling_type": self.labeling_type,
            "post_labeling_delay": self.post_labeling_delay,
            "labeling_duration": self.labeling_duration,
            "bolus_cutoff_flag": self.bolus_cutoff_flag,
            "bolus_cutoff_delay": self.bolus_cutoff_delay,
            "background_suppression": self.background_suppression,
            "bs_pulse_times": self.bs_pulse_times,
            "venc": self.venc,
            "m0_type": self.m0_type,
            "temporal_resolution": self.temporal_resolution,
            "slice_timing": self.slice_timing,
        }

        # Handle numpy arrays
        if self.b_values is not None:
            result["b_values"] = self.b_values.tolist()
        else:
            result["b_values"] = None

        if self.b_vectors is not None:
            result["b_vectors"] = self.b_vectors.tolist()
        else:
            result["b_vectors"] = None

        # Add extra fields
        result.update(self.extra)

        return result


class VendorParser(ABC):
    """Abstract base class for vendor-specific DICOM parsing.

    Subclasses implement vendor-specific extraction logic for
    Siemens, GE, and Philips scanners.

    Examples
    --------
    >>> parser = SiemensParser()
    >>> if parser.can_parse(dcm):
    ...     metadata = parser.extract_metadata(dcm)
    ...     print(metadata.labeling_type)
    'PCASL'
    """

    # Vendor name constant for subclasses
    VENDOR_NAME: str = ""

    @abstractmethod
    def can_parse(self, dcm: Any) -> bool:
        """Check if this parser can handle the given DICOM dataset.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset to check.

        Returns
        -------
        bool
            True if this parser can extract vendor-specific metadata.
        """
        ...

    @abstractmethod
    def extract_metadata(self, dcm: Any) -> VendorMetadata:
        """Extract vendor-specific metadata from DICOM dataset.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset to parse.

        Returns
        -------
        VendorMetadata
            Extracted metadata with vendor-specific parameters.
        """
        ...

    def _get_standard_metadata(self, dcm: Any) -> VendorMetadata:
        """Extract standard DICOM metadata common to all vendors.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset to parse.

        Returns
        -------
        VendorMetadata
            Metadata with standard DICOM fields populated.
        """
        metadata = VendorMetadata(vendor=self.VENDOR_NAME)

        # Software version
        if hasattr(dcm, "SoftwareVersions"):
            versions = dcm.SoftwareVersions
            if isinstance(versions, (list, tuple)):
                metadata.software_version = str(versions[0])
            else:
                metadata.software_version = str(versions)

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

        return metadata

    def _safe_get_private_tag(
        self, dcm: Any, group: int, element: int, default: Any = None
    ) -> Any:
        """Safely retrieve a private DICOM tag.

        Parameters
        ----------
        dcm : pydicom.Dataset
            DICOM dataset.
        group : int
            Tag group number (e.g., 0x0019).
        element : int
            Tag element number (e.g., 0x100c).
        default : Any
            Default value if tag not found.

        Returns
        -------
        Any
            Tag value or default.
        """
        tag = (group, element)
        if tag in dcm:
            elem = dcm[tag]
            return elem.value if elem.value is not None else default
        return default
