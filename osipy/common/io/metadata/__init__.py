"""Metadata mapping and validation for osipy.

This subpackage provides metadata extraction, mapping, validation,
and interactive prompting for missing parameters.

"""

from osipy.common.io.metadata.defaults import (
    DEFAULT_ASL_PARAMS,
    DEFAULT_DCE_PARAMS,
    DEFAULT_DSC_PARAMS,
    DEFAULT_IVIM_PARAMS,
    get_default_params,
)
from osipy.common.io.metadata.mapper import MetadataMapper
from osipy.common.io.metadata.prompter import ParameterPrompter
from osipy.common.io.metadata.validator import (
    REQUIRED_PARAMS,
    ParameterValidator,
    ValidationResult,
)

__all__ = [
    "DEFAULT_ASL_PARAMS",
    "DEFAULT_DCE_PARAMS",
    "DEFAULT_DSC_PARAMS",
    "DEFAULT_IVIM_PARAMS",
    "REQUIRED_PARAMS",
    "MetadataMapper",
    "ParameterPrompter",
    "ParameterValidator",
    "ValidationResult",
    "get_default_params",
]
