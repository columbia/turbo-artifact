import yaml
import math
import numpy as np
from privacypacking.cache.cache import Cache
from privacypacking.budget.block import HyperBlock
from privacypacking.budget.curves import LaplaceCurve, ZeroCurve


class CacheEntry:
    def __init__(self, result, noise_std, noise):
        self.result = result  # True result without noise
        self.noise_std = noise_std  # std of Laplace distribution (TODO: generalize for more distributions)
        self.noise = noise  # The actual noise sampled from the distribution


class DeterministicCache(Cache):
    def __init__(self, variance_reduction):
        self.key_values = {}
        self.variance_reduction = variance_reduction

    def add_entry(self, query_id, hyperblock_id, cache_entry):
        if query_id not in self.key_values:
            self.key_values[query_id] = {}
        self.key_values[query_id].update({hyperblock_id: cache_entry})

    def get_entry(self, query_id, hyperblock_id):
        if query_id in self.key_values:
            if hyperblock_id in self.key_values[query_id]:
                cache_entry = self.key_values[query_id][hyperblock_id]
                return cache_entry
        return None

    def run(self, query_id, query, noise_std, hyperblock: HyperBlock):
        """
        noise_std is the std of the noise that a sensitivity 1 query is willing to accept
        laplace_scale = 1/pure_epsilon and std = \sqrt{2} * laplace_scale
        For queries that have sensitivity != 1 we need to multiply the noise, but for now we just have counts.
        """
        # TODO: maybe use alpha, beta, or std instead of noise scale? And add sensitivity argument?

        run_metadata = {}
        cache_entry = self.get_entry(query_id, hyperblock.id)
        if cache_entry is None:  # Not cached
            # Obtain true result by running the query
            true_result = hyperblock.run(query)
            laplace_scale = noise_std / np.sqrt(2)
            run_budget = LaplaceCurve(laplace_noise=laplace_scale)
            noise = np.random.laplace(scale=laplace_scale)

        else:  # Cached
            true_result = cache_entry.result

            if noise_std >= cache_entry.noise_std:
                # We already have a good estimate in the cache
                run_budget = ZeroCurve()
                noise = cache_entry.noise
            else:
                # We need to improve on the cache
                if not self.variance_reduction:
                    # Just compute from scratch and pay for it
                    laplace_scale = noise_std / np.sqrt(2)
                    run_budget = LaplaceCurve(laplace_noise=laplace_scale)
                    noise = np.random.laplace(scale=laplace_scale)
                else:
                    # Var[X] = 2x^2, Y ∼ Lap(y). X might not follow a Laplace distribution!
                    # Var[aX + bY] = 2(ax)^2 + 2(by)^2 = c
                    # We set a ∈ [0,1] and b = 1-a
                    # Then, we maximize y^2 = f(a) = (c - 2(ax)^2)/2(1-a)^2
                    # We have (1-a)^3 f'(a) = c - 2ax^2
                    # So we take a = c/(2x^2)
                    x = cache_entry.noise_std / np.sqrt(2)
                    c = noise_std ** 2
                    a = c / (2 * (x ** 2))
                    b = 1 - a
                    y = np.sqrt((c - 2 * (a * x) ** 2) / (2 * b ** 2))

                    assert np.isclose(2 * (a * x) ** 2 + 2 * (b * y) ** 2, c)

                    # Get some fresh noise with optimal variance and take a linear combination with the old noise
                    laplace_scale = y / np.sqrt(2)
                    fresh_noise = np.random.laplace(scale=laplace_scale)
                    run_budget = LaplaceCurve(laplace_noise=laplace_scale)
                    noise = a * noise + b * fresh_noise

        # If we used any fresh noise we need to update the cache
        if not isinstance(run_budget, ZeroCurve):
            cache_entry = CacheEntry(
                result=true_result, noise_std=noise_std, noise=noise
            )
            self.add_entry(query_id, hyperblock.id, cache_entry)

        result = true_result + noise
        return result, run_budget, run_metadata

    def estimate_run_budget(self, query_id, hyperblock, noise_std):
        cache_entry = self.get_entry(query_id, hyperblock.id)
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
