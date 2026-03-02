"""Default parameter values for each modality.

This module provides default acquisition parameters when values
cannot be extracted from DICOM or BIDS sources.

References
----------
OSIPI White Papers for recommended parameter ranges
Alsop DC et al. (2015). Recommended implementation of ASL. MRM.
"""

from typing import Any

from osipy.common.types import Modality

# Default DCE parameters
DEFAULT_DCE_PARAMS: dict[str, Any] = {
    "tr": 5.0,  # ms - typical for fast 3D GRE
    "te": 2.0,  # ms
    "flip_angle": 15.0,  # degrees - typical for DCE
    "temporal_resolution": 5.0,  # seconds
    "baseline_frames": 5,
    "relaxivity": 4.5,  # mM^-1 s^-1 at 3T for Gd agents
}

# Default DSC parameters
DEFAULT_DSC_PARAMS: dict[str, Any] = {
    "tr": 1500.0,  # ms - typical for EPI
    "te": 30.0,  # ms - T2* weighted
    "temporal_resolution": 1.5,  # seconds
    "baseline_frames": 10,
}

# Default ASL parameters (based on ISMRM consensus recommendations)
DEFAULT_ASL_PARAMS: dict[str, Any] = {
    "tr": 4500.0,  # ms - typical for pCASL
    "te": 12.0,  # ms
    "labeling_type": "PCASL",  # Most common modern implementation
    "post_labeling_delay": 1800.0,  # ms - recommended for adult brain
    "labeling_duration": 1800.0,  # ms - recommended for pCASL
    "background_suppression": True,
    "bs_efficiency": 0.95,  # Typical efficiency
}

# Default IVIM parameters
DEFAULT_IVIM_PARAMS: dict[str, Any] = {
    "tr": 3000.0,  # ms
    "te": 70.0,  # ms - diffusion weighted
    "b_values": [0, 10, 20, 40, 80, 110, 140, 170, 200, 300, 400, 500, 600, 700, 800],
}

# Mapping of modality to default parameters
_MODALITY_DEFAULTS: dict[Modality, dict[str, Any]] = {
    Modality.DCE: DEFAULT_DCE_PARAMS,
    Modality.DSC: DEFAULT_DSC_PARAMS,
    Modality.ASL: DEFAULT_ASL_PARAMS,
    Modality.IVIM: DEFAULT_IVIM_PARAMS,
}


def get_default_params(modality: Modality) -> dict[str, Any]:
    """Get default parameters for a modality.

    Parameters
    ----------
    modality : Modality
        Perfusion imaging modality.

    Returns
    -------
    dict[str, Any]
        Dictionary of default parameter values.

    Examples
    --------
    >>> from osipy.common.types import Modality
    >>> defaults = get_default_params(Modality.ASL)
    >>> print(defaults["post_labeling_delay"])
    1800.0
    """
    return _MODALITY_DEFAULTS.get(modality, {}).copy()


def get_default_value(modality: Modality, param_name: str) -> Any:
    """Get a single default parameter value.

    Parameters
    ----------
    modality : Modality
        Perfusion imaging modality.
    param_name : str
        Parameter name.

    Returns
    -------
    Any
        Default value, or None if not defined.

    Examples
    --------
    >>> from osipy.common.types import Modality
    >>> pld = get_default_value(Modality.ASL, "post_labeling_delay")
    >>> print(pld)
    1800.0
    """
    defaults = _MODALITY_DEFAULTS.get(modality, {})
    return defaults.get(param_name)


# Human-readable descriptions for interactive prompting
PARAM_DESCRIPTIONS: dict[str, str] = {
    "tr": "Repetition Time in milliseconds",
    "te": "Echo Time in milliseconds",
    "ti": "Inversion Time in milliseconds",
    "flip_angle": "Flip angle in degrees",
    "field_strength": "Magnetic field strength in Tesla",
    "labeling_type": "ASL labeling type (PCASL, PASL, VSASL)",
    "post_labeling_delay": "Post-labeling delay in milliseconds",
    "labeling_duration": "Labeling duration in milliseconds",
    "background_suppression": "Whether background suppression was used (yes/no)",
    "b_values": "Diffusion b-values in s/mm² (comma-separated)",
    "temporal_resolution": "Time between dynamic volumes in seconds",
    "baseline_frames": "Number of baseline frames before contrast/labeling",
    "relaxivity": "Contrast agent relaxivity in mM⁻¹s⁻¹",
}


# Valid ranges for parameter validation
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "tr": (1.0, 20000.0),  # 1ms to 20s
    "te": (0.1, 500.0),  # 0.1ms to 500ms
    "ti": (0.0, 10000.0),  # 0 to 10s
    "flip_angle": (0.1, 180.0),  # 0.1 to 180 degrees
    "field_strength": (0.1, 20.0),  # 0.1T to 20T
    "post_labeling_delay": (100.0, 10000.0),  # 100ms to 10s
    "labeling_duration": (100.0, 5000.0),  # 100ms to 5s
    "temporal_resolution": (0.1, 60.0),  # 0.1s to 60s
    "baseline_frames": (1.0, 100.0),  # 1 to 100 frames
    "relaxivity": (1.0, 20.0),  # 1 to 20 mM^-1 s^-1
}
