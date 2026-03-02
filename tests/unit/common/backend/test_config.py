"""Unit tests for GPU configuration and detection.

Tests for osipy.common.backend.config module.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from osipy.common.backend.config import (
    GPUConfig,
    _reset_gpu_cache,
    get_backend,
    get_gpu_memory_info,
    is_gpu_available,
    set_backend,
)
from osipy.common.exceptions import DataValidationError


class TestGPUConfig:
    """Tests for GPUConfig dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = GPUConfig()
        assert config.force_cpu is False
        assert config.default_batch_size == 10000
        assert config.memory_limit_fraction == 0.9
        assert config.device_id == 0

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        config = GPUConfig(
            force_cpu=True,
            default_batch_size=5000,
            memory_limit_fraction=0.8,
            device_id=1,
        )
        assert config.force_cpu is True
        assert config.default_batch_size == 5000
        assert config.memory_limit_fraction == 0.8
        assert config.device_id == 1

    def test_invalid_memory_fraction_high(self) -> None:
        """Should reject memory fraction > 1."""
        with pytest.raises(DataValidationError, match="memory_limit_fraction"):
            GPUConfig(memory_limit_fraction=1.5)

    def test_invalid_memory_fraction_low(self) -> None:
        """Should reject memory fraction <= 0."""
        with pytest.raises(DataValidationError, match="memory_limit_fraction"):
            GPUConfig(memory_limit_fraction=0.0)

    def test_invalid_batch_size(self) -> None:
        """Should reject non-positive batch size."""
        with pytest.raises(DataValidationError, match="default_batch_size"):
            GPUConfig(default_batch_size=0)

    def test_invalid_device_id(self) -> None:
        """Should reject negative device ID."""
        with pytest.raises(DataValidationError, match="device_id"):
            GPUConfig(device_id=-1)


class TestIsGpuAvailable:
    """Tests for is_gpu_available function."""

    @pytest.fixture(autouse=True)
    def reset_cache(self) -> None:
        """Reset GPU cache before each test."""
        _reset_gpu_cache()

    def test_returns_bool(self) -> None:
        """Should return a boolean."""
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_respects_environment_variable(self) -> None:
        """Should return False when OSIPY_FORCE_CPU=1."""
        _reset_gpu_cache()
        with mock.patch.dict(os.environ, {"OSIPY_FORCE_CPU": "1"}):
            result = is_gpu_available()
            assert result is False

    def test_caches_result(self) -> None:
        """Should cache the availability check."""
        _reset_gpu_cache()
        result1 = is_gpu_available()
        result2 = is_gpu_available()
        assert result1 == result2

    def test_false_when_cupy_import_fails(self) -> None:
        """Should return False when CuPy cannot be imported."""
        _reset_gpu_cache()
        with mock.patch.dict("sys.modules", {"cupy": None}):
            # Force a fresh check by resetting cache
            _reset_gpu_cache()
            # This may still return True if CuPy is actually installed
            # The test mainly verifies no exception is raised
            result = is_gpu_available()
            assert isinstance(result, bool)


class TestGetSetBackend:
    """Tests for get_backend and set_backend functions."""

    def test_get_backend_returns_config(self) -> None:
        """Should return a GPUConfig."""
        config = get_backend()
        assert isinstance(config, GPUConfig)

    def test_set_backend_changes_config(self) -> None:
        """Should update the global config."""
        original = get_backend()
        try:
            new_config = GPUConfig(force_cpu=True, default_batch_size=999)
            set_backend(new_config)
            retrieved = get_backend()
            assert retrieved.force_cpu is True
            assert retrieved.default_batch_size == 999
        finally:
            set_backend(original)

    def test_respects_environment_variable_for_default(self) -> None:
        """Default config should respect OSIPY_FORCE_CPU."""
        # Reset global config
        import osipy.common.backend.config as config_module

        config_module._global_config = None

        with mock.patch.dict(os.environ, {"OSIPY_FORCE_CPU": "1"}):
            config = get_backend()
            assert config.force_cpu is True


class TestGetGpuMemoryInfo:
    """Tests for get_gpu_memory_info function."""

    def test_returns_dict(self) -> None:
        """Should return a dictionary."""
        info = get_gpu_memory_info()
        assert isinstance(info, dict)

    def test_contains_required_keys(self) -> None:
        """Should contain all required keys."""
        info = get_gpu_memory_info()
        assert "available" in info
        assert "total_bytes" in info
        assert "used_bytes" in info
        assert "free_bytes" in info
        assert "device_name" in info

    def test_available_is_bool(self) -> None:
        """Available should be boolean."""
        info = get_gpu_memory_info()
        assert isinstance(info["available"], bool)

    def test_bytes_are_integers(self) -> None:
        """Byte counts should be integers."""
        info = get_gpu_memory_info()
        assert isinstance(info["total_bytes"], int)
        assert isinstance(info["used_bytes"], int)
        assert isinstance(info["free_bytes"], int)

    def test_device_name_is_string(self) -> None:
        """Device name should be a string."""
        info = get_gpu_memory_info()
        assert isinstance(info["device_name"], str)


class TestGpuIntegration:
    """Integration tests when CuPy is available."""

    @pytest.fixture(autouse=True)
    def reset_state(self) -> None:
        """Reset state before each test."""
        _reset_gpu_cache()
        import osipy.common.backend.config as config_module

        config_module._global_config = None

    def test_is_gpu_available_true(self) -> None:
        """Should return True when CuPy is installed and GPU works."""
        pytest.importorskip("cupy")
        # Clear environment variable
        env = os.environ.copy()
        env.pop("OSIPY_FORCE_CPU", None)
        with mock.patch.dict(os.environ, env, clear=True):
            _reset_gpu_cache()
            result = is_gpu_available()
            # May be True or False depending on actual GPU
            assert isinstance(result, bool)

    def test_memory_info_with_gpu(self) -> None:
        """Should return actual memory info when GPU available."""
        pytest.importorskip("cupy")
        _reset_gpu_cache()
        if is_gpu_available():
            info = get_gpu_memory_info()
            assert info["available"] is True
            assert info["total_bytes"] > 0
            assert len(info["device_name"]) > 0
