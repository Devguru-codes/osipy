"""Tests for concentration model registry."""

import pytest

from osipy.common.exceptions import DataValidationError
from osipy.dce.concentration.registry import (
    get_concentration_model,
    list_concentration_models,
)


class TestConcentrationRegistry:
    @pytest.mark.parametrize("name", list_concentration_models())
    def test_get_model(self, name):
        func = get_concentration_model(name)
        assert callable(func)

    def test_unknown_raises(self):
        with pytest.raises(DataValidationError, match="Unknown concentration model"):
            get_concentration_model("unknown")

    def test_list_sorted(self):
        result = list_concentration_models()
        assert result == sorted(result)
