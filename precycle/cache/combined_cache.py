from precycle.cache.cache import Cache
from precycle.cache.deterministic_cache import MockDeterministicCache
from precycle.cache.probabilistic_cache import MockProbabilisticCache


class CombinedCache(Cache):
    def __init__(self, config):
        pass


class MockCombinedCache(Cache):
    def __init__(self, config):
        self.config = config
        self.deterministic_cache = MockDeterministicCache(config)
        self.probabilistic_cache = MockProbabilisticCache(config)
