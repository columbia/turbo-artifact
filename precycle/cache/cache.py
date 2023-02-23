from precycle.cache.laplace_cache import MockLaplaceCache
from precycle.cache.histogram_cache import MockHistogramCache
from precycle.cache.pmw_cache import MockPMWCache
from precycle.cache.sparse_vectors import MockSparseVectors


class Cache:
    def __init__(self, config):
        self.cache_type = config.cache.type


class MockCache:
    def __init__(self, config):
        self.config = config
        self.cache_type = config.cache.type

        if self.cache_type == "LaplaceCache":
            self.laplace_cache = MockLaplaceCache(config)
        elif self.cache_type == "PMWCache":
            self.pmw_cache = MockPMWCache(config)
        elif self.cache_type == "HybridCache":
            self.laplace_cache = MockLaplaceCache(config)
            self.histogram_cache = MockHistogramCache(config)
            self.sparse_vectors = MockSparseVectors(config)
