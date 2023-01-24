import yaml
import math
import redis
from precycle.cache.cache import Cache
from precycle.budget.curves import LaplaceCurve, ZeroCurve
from precycle.utils.utils import get_blocks_size


class CacheEntry:
    def __init__(self, result, noise_std, noise):
        self.result = result  # True result without noise
        self.noise_std = noise_std  # std of Laplace distribution
        self.noise = noise  # The actual noise sampled from the distribution


class CacheKey:
    def __init__(self, query_id, blocks):
        self.key = f"{query_id}:{blocks}"


class DeterministicCache(Cache):
    def __init__(self, config):
        self.kv_store = redis.Redis(
            host=config.cache.host, port=config.cache.port, db=0
        )

    def add_entry(self, query_id, blocks, cache_entry):
        key = CacheKey(query_id, blocks).key
        self.kv_store.hset(key, "result", cache_entry.result)
        self.kv_store.hset(key, "noise_std", cache_entry.noise_std)
        self.kv_store.hset(key, "noise", cache_entry.noise)

    def get_entry(self, query_id, blocks):
        key = CacheKey(query_id, blocks).key
        entry = self.kv_store.hgetall(key)
        entry_values = [float(value) for value in entry.values()]
        if entry:
            return CacheEntry(entry_values[0], entry_values[1], entry_values[2])
        return None

    def estimate_run_budget(self, query_id, blocks, noise_std):
        """
        Checks the cache and returns the budget we need to spend to reach the desired 'noise_std'
        """
        cache_entry = self.get_entry(query_id, blocks)
        if cache_entry is not None:
            if noise_std >= cache_entry.noise_std:
                # Good enough estimate
                return ZeroCurve()

        # TODO: re-enable variance reduction
        laplace_scale = noise_std / math.sqrt(2)
        # Budget doesn't care about sensitivity
        sensitivity = 1 / get_blocks_size(blocks, self.config.blocks_metadata)
        run_budget = LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
        return run_budget

    def dump(self):
        pass


class MockDeterministicCache(Cache):
    def __init__(self, config):
        # key-value store is just an in-memory dictionary
        self.config = config
        self.kv_store = {}

    def add_entry(self, query_id, blocks, cache_entry):
        key = CacheKey(query_id, blocks).key
        self.kv_store[key] = {
            "result": cache_entry.result,
            "noise_std": cache_entry.noise_std,
            "noise": cache_entry.noise,
        }

    def get_entry(self, query_id, blocks):
        key = CacheKey(query_id, blocks).key
        if key in self.kv_store:
            entry = self.kv_store[key]
            return CacheEntry(entry["result"], entry["noise_std"], entry["noise"])
        return None

    def estimate_run_budget(self, query_id, blocks, noise_std):
        """
        Checks the cache and returns the budget we need to spend to reach the desired 'noise_std'
        """
        cache_entry = self.get_entry(query_id, blocks)
        if cache_entry is not None:
            if noise_std >= cache_entry.noise_std:
                # Good enough estimate
                return ZeroCurve()

        # TODO: re-enable variance reduction
        laplace_scale = noise_std / math.sqrt(2)
        # Budget doesn't care about sensitivity
        sensitivity = 1 / get_blocks_size(blocks, self.config.blocks_metadata)
        run_budget = LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
        return run_budget

    def dump(self):
        print("Cache", yaml.dump(self.key_values))
