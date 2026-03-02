"""Unit tests for intermediate result caching.

Tests for osipy/common/caching.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from osipy.common.caching import (
    CacheConfig,
    IntermediateCache,
    RetentionPolicy,
    configure_cache,
    get_cache,
)


class TestRetentionPolicy:
    """Tests for RetentionPolicy enum."""

    def test_policy_values(self) -> None:
        """Test retention policy values."""
        assert RetentionPolicy.TRANSIENT.value == "transient"
        assert RetentionPolicy.CACHED.value == "cached"
        assert RetentionPolicy.PERSISTENT.value == "persistent"


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CacheConfig()
        assert config.default_policy == RetentionPolicy.CACHED
        assert config.policies == {}
        assert config.cache_dir is None
        assert config.max_memory_mb == 1024
        assert config.max_age_seconds == 3600
        assert config.compression is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CacheConfig(
            default_policy=RetentionPolicy.PERSISTENT,
            policies={"t1_map": RetentionPolicy.TRANSIENT},
            max_memory_mb=512,
            max_age_seconds=1800,
        )
        assert config.default_policy == RetentionPolicy.PERSISTENT
        assert config.policies["t1_map"] == RetentionPolicy.TRANSIENT


class TestIntermediateCache:
    """Tests for IntermediateCache class."""

    @pytest.fixture
    def cache(self) -> IntermediateCache:
        """Create cache for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(cache_dir=Path(tmpdir))
            yield IntermediateCache(config)

    def test_put_and_get(self, cache: IntermediateCache) -> None:
        """Test basic put and get operations."""
        data = np.random.rand(10, 10, 5)

        cache.put("t1_map", "test_session", data)
        retrieved = cache.get("t1_map", "test_session")

        np.testing.assert_array_equal(retrieved, data)

    def test_get_nonexistent(self, cache: IntermediateCache) -> None:
        """Test getting non-existent key returns None."""
        result = cache.get("t1_map", "nonexistent")
        assert result is None

    def test_has_method(self, cache: IntermediateCache) -> None:
        """Test has method."""
        data = np.random.rand(5, 5, 3)

        assert not cache.has("t1_map", "test")
        cache.put("t1_map", "test", data)
        assert cache.has("t1_map", "test")

    def test_invalidate(self, cache: IntermediateCache) -> None:
        """Test invalidating cached data."""
        data = np.random.rand(5, 5, 3)

        cache.put("t1_map", "test", data)
        assert cache.has("t1_map", "test")

        cache.invalidate("t1_map", "test")
        assert not cache.has("t1_map", "test")

    def test_clear_all(self, cache: IntermediateCache) -> None:
        """Test clearing all cached data."""
        cache.put("t1_map", "test1", np.random.rand(5, 5, 3))
        cache.put("concentration", "test2", np.random.rand(5, 5, 3))

        cache.clear()

        assert not cache.has("t1_map", "test1")
        assert not cache.has("concentration", "test2")

    def test_clear_by_type(self, cache: IntermediateCache) -> None:
        """Test clearing by result type."""
        cache.put("t1_map", "test1", np.random.rand(5, 5, 3))
        cache.put("concentration", "test2", np.random.rand(5, 5, 3))

        cache.clear("t1_map")

        assert not cache.has("t1_map", "test1")
        assert cache.has("concentration", "test2")

    def test_transient_policy_not_cached(self) -> None:
        """Test transient policy doesn't cache data."""
        config = CacheConfig(policies={"t1_map": RetentionPolicy.TRANSIENT})
        cache = IntermediateCache(config)

        data = np.random.rand(5, 5, 3)
        cache.put("t1_map", "test", data)

        assert not cache.has("t1_map", "test")

    def test_persistent_policy_saves_to_disk(self) -> None:
        """Test persistent policy saves to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(
                cache_dir=Path(tmpdir),
                policies={"t1_map": RetentionPolicy.PERSISTENT},
            )
            cache = IntermediateCache(config)

            data = np.random.rand(5, 5, 3)
            cache.put("t1_map", "test", data)

            # Check file exists
            cache_files = list(Path(tmpdir).glob("*.npz"))
            assert len(cache_files) >= 1

    def test_memory_eviction(self) -> None:
        """Test memory eviction when limit reached."""
        config = CacheConfig(max_memory_mb=1)  # 1 MB limit
        cache = IntermediateCache(config)

        # Add data that exceeds limit
        for i in range(10):
            # Each array is ~1 MB
            data = np.random.rand(128, 128, 10)
            cache.put("data", f"test_{i}", data)

        stats = cache.get_stats()
        # Memory should be limited
        assert stats["memory_size_mb"] <= config.max_memory_mb + 1

    def test_get_stats(self, cache: IntermediateCache) -> None:
        """Test getting cache statistics."""
        data = np.random.rand(10, 10, 5)
        cache.put("t1_map", "test", data)

        stats = cache.get_stats()

        assert "memory_entries" in stats
        assert "memory_size_mb" in stats
        assert "max_memory_mb" in stats
        assert "cache_dir" in stats

        assert stats["memory_entries"] == 1

    def test_dict_data_storage(self, cache: IntermediateCache) -> None:
        """Test storing dict data."""
        data = {
            "param1": np.random.rand(5, 5, 3),
            "param2": np.random.rand(5, 5, 3),
        }

        cache.put("params", "test", data)
        retrieved = cache.get("params", "test")

        assert isinstance(retrieved, dict)
        np.testing.assert_array_equal(retrieved["param1"], data["param1"])

    def test_metadata_storage(self, cache: IntermediateCache) -> None:
        """Test storing metadata with data."""
        data = np.random.rand(5, 5, 3)
        metadata = {"source": "test", "version": 1}

        cache.put("t1_map", "test", data, metadata=metadata)
        # Metadata is stored but not returned by get
        retrieved = cache.get("t1_map", "test")

        assert retrieved is not None


class TestGlobalCache:
    """Tests for global cache functions."""

    def test_get_cache_singleton(self) -> None:
        """Test get_cache returns singleton."""
        cache1 = get_cache()
        cache2 = get_cache()

        assert cache1 is cache2

    def test_configure_cache(self) -> None:
        """Test configuring global cache."""
        config = CacheConfig(max_memory_mb=512)
        configure_cache(config)

        cache = get_cache()
        stats = cache.get_stats()

        assert stats["max_memory_mb"] == 512


class TestCacheWithPolicies:
    """Tests for cache with different policies per type."""

    def test_per_type_policies(self) -> None:
        """Test different policies for different types."""
        config = CacheConfig(
            default_policy=RetentionPolicy.CACHED,
            policies={
                "t1_map": RetentionPolicy.PERSISTENT,
                "temp": RetentionPolicy.TRANSIENT,
            },
        )
        cache = IntermediateCache(config)

        assert cache.get_policy("t1_map") == RetentionPolicy.PERSISTENT
        assert cache.get_policy("temp") == RetentionPolicy.TRANSIENT
        assert cache.get_policy("other") == RetentionPolicy.CACHED
