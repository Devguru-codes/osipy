"""Tests for T1 mapping method registry."""

import pytest

from osipy.common.exceptions import DataValidationError
from osipy.dce.t1_mapping.registry import get_t1_method, list_t1_methods


class TestT1Registry:
    @pytest.mark.parametrize("name", list_t1_methods())
    def test_get_method(self, name):
        func = get_t1_method(name)
        assert callable(func)

    def test_unknown_raises(self):
        with pytest.raises(DataValidationError, match="Unknown T1 method"):
            get_t1_method("unknown")

    def test_list_sorted(self):
        result = list_t1_methods()
        assert result == sorted(result)
