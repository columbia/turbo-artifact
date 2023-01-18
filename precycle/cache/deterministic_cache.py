import yaml
import math
import redis
from cache.cache import Cache
from budget.curves import LaplaceCurve, ZeroCurve


class CacheEntry:
    def __init__(self, result, noise_std, noise):
        self.result = result  # True result without noise
        self.noise_std = noise_std  # std of Laplace distribution
        self.noise = noise  # The actual noise sampled from the distribution


class CacheKey:
    def __init__(self, query_id, hyperblock_id):
        self.key = f"{query_id} : {hyperblock_id}"


class DeterministicCache(Cache):
    def __init__(self, config):
        self.kv_store = redis.Redis(host=config.host, port=config.port, db=0)

    def add_entry(self, query_id, hyperblock_id, cache_entry):
        key = CacheKey(query_id, hyperblock_id).key
        self.kv_store.hset(key, "result", cache_entry.result)
        self.kv_store.hset(key, "noise_std", cache_entry.noise_std)
        self.kv_store.hset(key, "noise", cache_entry.noise)

    def get_entry(self, query_id, hyperblock_id):
        key = CacheKey(query_id, hyperblock_id).key
        entry = self.kv_store.hgetall(key)
        cache_entry = CacheEntry(entry["result"], entry["noise_std"], entry["noise"])
        return cache_entry

    def estimate_run_budget(self, query_id, hyperblock_id, noise_std):
        """
        Checks the cache and returns the budget we need to spend to reach the desired 'noise_std'
        """
        cache_entry = self.get_entry(query_id, hyperblock_id)
        if cache_entry is not None:
            if noise_std >= cache_entry.noise_std:
                # Good enough estimate
                return ZeroCurve()

        # TODO: re-enable variance reduction
        laplace_scale = noise_std / math.sqrt(2)
        run_budget = LaplaceCurve(laplace_noise=laplace_scale)
        return run_budget

    def dump(self):
        print("Cache", yaml.dump(self.key_values))
