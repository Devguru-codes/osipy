"""Tests for AIF detector registry."""

import pytest

from osipy.common.aif.base_detector import BaseAIFDetector
from osipy.common.aif.detection_registry import get_aif_detector, list_aif_detectors
from osipy.common.exceptions import DataValidationError


class TestAIFDetectorRegistry:
    @pytest.mark.parametrize("name", list_aif_detectors())
    def test_get_detector(self, name):
        detector = get_aif_detector(name)
        assert isinstance(detector, BaseAIFDetector)

    def test_unknown_raises(self):
        with pytest.raises(DataValidationError, match="Unknown AIF detector"):
            get_aif_detector("unknown")

    def test_list_sorted(self):
        result = list_aif_detectors()
        assert result == sorted(result)
