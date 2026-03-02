"""Interactive parameter prompting for missing metadata.

This module provides interactive prompting functionality for
acquiring missing acquisition parameters from the user.

"""

import logging
import sys
from typing import Any

from osipy.common.io.metadata.defaults import (
    PARAM_DESCRIPTIONS,
    PARAM_RANGES,
    get_default_value,
)
from osipy.common.types import Modality

logger = logging.getLogger(__name__)


class ParameterPrompter:
    """Interactive prompting for missing acquisition parameters.

    This class provides a consistent interface for prompting users
    to provide missing parameters, with type validation and default
    value suggestions.

    Examples
    --------
    >>> from osipy.common.types import Modality
    >>> prompter = ParameterPrompter(Modality.ASL, interactive=True)
    >>> value = prompter.prompt_for_value("post_labeling_delay")
    Missing post_labeling_delay (Post-labeling delay in milliseconds)
    Default: 1800.0
    Enter value [1800.0]:
    """

    def __init__(
        self,
        modality: Modality,
        interactive: bool = True,
        use_defaults: bool = True,
    ):
        """Initialize the prompter.

        Parameters
        ----------
        modality : Modality
            Perfusion imaging modality (for default values).
        interactive : bool, default=True
            Whether to prompt interactively. If False, returns defaults.
        use_defaults : bool, default=True
            Whether to use default values when available.
        """
        self.modality = modality
        self.interactive = interactive
        self.use_defaults = use_defaults

    def prompt_for_value(
        self,
        param_name: str,
        expected_type: type = float,
        default: Any = None,
    ) -> Any:
        """Prompt user for a missing parameter value.

        Parameters
        ----------
        param_name : str
            Name of the parameter.
        expected_type : type
            Expected type of the value.
        default : Any, optional
            Default value to suggest. If None, uses modality default.

        Returns
        -------
        Any
            User-provided value, default value, or None.
        """
        # Get default from modality if not provided
        if default is None and self.use_defaults:
            default = get_default_value(self.modality, param_name)

        # Non-interactive mode: return default
        if not self.interactive:
            if default is not None:
                logger.info(f"Using default {param_name}={default}")
            return default

        # Check if stdin is available for interactive input
        if not sys.stdin.isatty():
            logger.warning(
                f"Non-interactive environment, using default for {param_name}"
            )
            return default

        # Get description
        description = PARAM_DESCRIPTIONS.get(param_name, param_name)

        # Format prompt
        prompt_lines = [f"\nMissing {param_name} ({description})"]

        # Add range info if available
        if param_name in PARAM_RANGES:
            min_val, max_val = PARAM_RANGES[param_name]
            prompt_lines.append(f"Valid range: [{min_val}, {max_val}]")

        if default is not None:
            prompt_lines.append(f"Default: {default}")
            prompt = f"Enter value [{default}]: "
        else:
            prompt = "Enter value: "

        print("\n".join(prompt_lines))

        try:
            user_input = input(prompt).strip()

            # Empty input uses default
            if not user_input:
                return default

            # Parse based on expected type
            return self._parse_value(user_input, expected_type, param_name)

        except (EOFError, KeyboardInterrupt):
            logger.warning(f"Input interrupted, using default for {param_name}")
            return default

    def _parse_value(self, value_str: str, expected_type: type, param_name: str) -> Any:
        """Parse string value to expected type.

        Parameters
        ----------
        value_str : str
            String value from user input.
        expected_type : type
            Expected type.
        param_name : str
            Parameter name (for special handling).

        Returns
        -------
        Any
            Parsed value.
        """
        # Special handling for certain parameters
        if param_name == "labeling_type":
            return self._parse_labeling_type(value_str)
        elif param_name == "b_values":
            return self._parse_b_values(value_str)
        elif param_name == "background_suppression":
            return self._parse_bool(value_str)

        # Standard type conversion
        try:
            if expected_type is float:
                return float(value_str)
            elif expected_type is int:
                return int(value_str)
            elif expected_type is bool:
                return self._parse_bool(value_str)
            else:
                return expected_type(value_str)
        except ValueError:
            logger.warning(f"Could not parse '{value_str}' as {expected_type.__name__}")
            return None

    def _parse_labeling_type(self, value_str: str) -> str:
        """Parse ASL labeling type string.

        Parameters
        ----------
        value_str : str
            User input string.

        Returns
        -------
        str
            Normalized labeling type.
        """
        value_upper = value_str.upper().strip()

        # Map common variants
        if value_upper in ["PCASL", "PSEUDO-CONTINUOUS", "PSEUDOCONTINUOUS"]:
            return "PCASL"
        elif value_upper in ["PASL", "PULSED"]:
            return "PASL"
        elif value_upper in ["CASL", "CONTINUOUS"]:
            return "CASL"
        elif value_upper in ["VSASL", "VELOCITY-SELECTIVE", "VS"]:
            return "VSASL"
        elif value_upper in ["FAIR"]:
            return "PASL"  # FAIR is a type of PASL
        else:
            return value_str.upper()

    def _parse_b_values(self, value_str: str) -> list[float]:
        """Parse b-values from comma-separated string.

        Parameters
        ----------
        value_str : str
            Comma-separated b-values.

        Returns
        -------
        list[float]
            List of b-values.
        """
        import numpy as np

        # Handle various separators
        value_str = value_str.replace(";", ",").replace(" ", ",")
        parts = [p.strip() for p in value_str.split(",") if p.strip()]

        try:
            b_values = [float(p) for p in parts]
            return np.array(sorted(b_values))
        except ValueError:
            logger.warning(f"Could not parse b-values from '{value_str}'")
            return None

    def _parse_bool(self, value_str: str) -> bool:
        """Parse boolean from string.

        Parameters
        ----------
        value_str : str
            User input.

        Returns
        -------
        bool
            Parsed boolean.
        """
        return value_str.lower() in ["true", "yes", "y", "1", "on"]

    def prompt_for_missing(
        self,
        current_params: dict[str, Any],
        missing_params: list[str],
    ) -> dict[str, Any]:
        """Prompt for all missing parameters.

        Parameters
        ----------
        current_params : dict[str, Any]
            Current parameter values.
        missing_params : list[str]
            List of missing parameter names.

        Returns
        -------
        dict[str, Any]
            Updated parameters with user-provided values.
        """
        updated = current_params.copy()

        for param_name in missing_params:
            expected_type = self._get_expected_type(param_name)
            value = self.prompt_for_value(param_name, expected_type)
            if value is not None:
                updated[param_name] = value

        return updated

    def _get_expected_type(self, param_name: str) -> type:
        """Get expected type for a parameter.

        Parameters
        ----------
        param_name : str
            Parameter name.

        Returns
        -------
        type
            Expected type.
        """
        # Map parameters to types
        type_map = {
            "tr": float,
            "te": float,
            "ti": float,
            "flip_angle": float,
            "field_strength": float,
            "labeling_type": str,
            "post_labeling_delay": float,
            "labeling_duration": float,
            "background_suppression": bool,
            "b_values": list,
            "temporal_resolution": float,
            "baseline_frames": int,
            "relaxivity": float,
        }
        return type_map.get(param_name, str)
