from precycle.cache.pmw_cache import MockPMWCache
from precycle.cache.sparse_vectors import MockSparseVectors, SparseVectors
from precycle.cache.laplace_cache import MockLaplaceCache, LaplaceCache
from precycle.cache.histogram_cache import MockHistogramCache, HistogramCache


class Cache:
    def __init__(self, config):
        self.config = config
        self.cache_type = config.cache.type

        if self.cache_type == "LaplaceCache":
            self.laplace_cache = LaplaceCache(config)
        elif self.cache_type == "HybridCache":
            self.laplace_cache = LaplaceCache(config)
            self.histogram_cache = HistogramCache(config)
            self.sparse_vectors = SparseVectors(config)


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
