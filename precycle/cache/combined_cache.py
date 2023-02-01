from precycle.cache.cache import Cache
from precycle.cache.deterministic_cache import MockDeterministicCache
from precycle.cache.probabilistic_cache import MockProbabilisticCache


class CombinedCache(Cache):
    def __init__(self, config):
        self.cache_type = config.cache.type


class MockCombinedCache(Cache):
    def __init__(self, config):
        self.config = config
        self.cache_type = config.cache.type

        if self.cache_type in {"DeterministicCache", "CombinedCache"}:
            self.deterministic_cache = MockDeterministicCache(config)
        if self.cache_type in {"ProbabilisticCache", "CombinedCache"}:
            self.probabilistic_cache = MockProbabilisticCache(config)
