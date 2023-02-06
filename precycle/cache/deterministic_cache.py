import yaml
import math
import redis
import numpy as np
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

    def estimate_run_budget(self, query_id, query, blocks, noise_std):
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

    def update_entry(self, query_id, blocks, true_result, noise_std, noise):
        cache_entry = self.get_entry(query_id, blocks)
        sensitivity = 1 / get_blocks_size(blocks, self.config.blocks_metadata)

        if not cache_entry:
            # If not Cached then update
            new_cache_entry = CacheEntry(
                result=true_result, noise_std=noise_std, noise=noise
            )
            self.add_entry(query_id, blocks, new_cache_entry)
        else:  # Cached
            if not self.config.variance_reduction:
                if cache_entry.noise_std > noise_std:
                    # If not VR then update only if the entry cached is worse than the new one
                    new_cache_entry = CacheEntry(
                        result=true_result, noise_std=noise_std, noise=noise
                    )
                    self.add_entry(query_id, blocks, new_cache_entry)
            else:
            #     # If VR update no matter what - we can use whatever is in the cache to get even better
                # TODO a temporary hack to enable VR. 
                cached_laplace_scale = cache_entry.noise_std / np.sqrt(2)
                cached_pure_epsilon = sensitivity / cached_laplace_scale
                
                incoming_laplace_scale = noise_std / np.sqrt(2)
                incoming_pure_epsilon = sensitivity / incoming_laplace_scale

                new_pure_epsilon = cached_pure_epsilon + incoming_pure_epsilon
                new_laplace_scale = sensitivity / new_pure_epsilon
                new_noise_std = new_laplace_scale * np.sqrt(2)
                # TODO: Temporary hack is that I don't compute the new noise by aggregating but resampling
                new_noise = np.random.laplace(scale=new_laplace_scale)
                new_cache_entry = CacheEntry(
                    result=true_result, noise_std=new_noise_std, noise=new_noise
                )
                self.add_entry(query_id, blocks, new_cache_entry)
            

    def estimate_run_budget(self, query_id, blocks, noise_std):
        """
        Checks the cache and returns the budget we need to spend to reach the desired 'noise_std'
        """
        # variance_reduction = False  # No VR for now
        cache_entry = self.get_entry(query_id, blocks)
        sensitivity = 1 / get_blocks_size(blocks, self.config.blocks_metadata)

        if not cache_entry:
            laplace_scale = noise_std / math.sqrt(2)
            run_budget = LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
        else:
            if noise_std >= cache_entry.noise_std:
                # Good enough estimate
                run_budget = ZeroCurve()
            else:
                if not self.config.variance_reduction:
                    laplace_scale = noise_std / math.sqrt(2)
                    sensitivity = 1 / get_blocks_size(
                        blocks, self.config.blocks_metadata
                    )
                    run_budget = LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
                else:
                    cached_laplace_scale = cache_entry.noise_std / np.sqrt(2)
                    cached_pure_epsilon = sensitivity / cached_laplace_scale

                    target_laplace_scale = noise_std / np.sqrt(2)
                    target_pure_epsilon = sensitivity / target_laplace_scale

                    run_pure_epsilon = target_pure_epsilon - cached_pure_epsilon
                    run_laplace_scale = sensitivity / run_pure_epsilon

                    run_budget = LaplaceCurve(laplace_noise=run_laplace_scale / sensitivity)

        return run_budget

    def dump(self):
        print("Cache", yaml.dump(self.key_values))
