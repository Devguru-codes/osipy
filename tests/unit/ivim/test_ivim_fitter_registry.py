"""Tests for IVIM fitter registry."""

import pytest

from osipy.common.exceptions import DataValidationError
from osipy.ivim.fitting.registry import get_ivim_fitter, list_ivim_fitters


class TestIVIMFitterRegistry:
    @pytest.mark.parametrize("name", list_ivim_fitters())
    def test_get_fitter(self, name):
        func = get_ivim_fitter(name)
        assert callable(func)

    def test_unknown_raises(self):
        with pytest.raises(DataValidationError, match="Unknown IVIM fitter"):
            get_ivim_fitter("unknown")

    def test_list_sorted(self):
        result = list_ivim_fitters()
        assert result == sorted(result)
