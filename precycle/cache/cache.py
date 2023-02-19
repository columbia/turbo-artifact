from precycle.cache.deterministic_cache import MockDeterministicCache
from precycle.cache.probabilistic_cache import MockProbabilisticCache
from precycle.cache.sparse_vectors import MockSparseVectors


class Cache:
    def __init__(self, config):
        self.cache_type = config.cache.type


class MockCache:
    def __init__(self, config):
        self.config = config
        self.cache_type = config.cache.type

        if self.cache_type in {"DeterministicCache", "CombinedCache"}:
            self.deterministic_cache = MockDeterministicCache(config)
        if self.cache_type in {"ProbabilisticCache", "CombinedCache"}:
            self.probabilistic_cache = MockProbabilisticCache(config)
            self.sparse_vectors = MockSparseVectors(config)
