import numpy as np
from typing import Dict, Tuple
from precycle.budget.curves import LaplaceCurve, ZeroCurve
from precycle.cache.deterministic_cache import CacheEntry


class R:
    def __init__(self, blocks, noise_std, cache_type) -> None:
        self.blocks = blocks
        self.noise_std = noise_std
        self.cache_type = cache_type

    def __str__(self):
        return f"Run({self.blocks}, {self.noise_std})"


class A:
    def __init__(self, l, cost=None) -> None:
        self.l = l
        self.cost = cost

    def __str__(self):
        return f"Aggregate({[str(l) for l in self.l]})"


class Executor:
    def __init__(self, cache, db) -> None:
        self.db = db
        self.cache = cache

    def execute_plan(self, plan, task) -> Tuple[float, Dict]:
        """
        run_budget: the budget that will be consumed from the blocks after running the query
        """
        result = None
        total_size = 0
        run_metadata = {}
        run_budget_per_block = {}

        results = []
        for run_op in plan.l:
            if run_op.cache_type == "deterministic":
                result, blocks_size, run_budget, run_metadata = self.run_deterministic(
                    run_op, task.query_id, task.query, run_metadata
                )
            elif run_op.cache_type == "probabilistic":
                result, blocks_size, run_budget, run_metadata = self.run_probabilistic(
                    run_op, task.query_id, task.query, run_metadata
                )
            run_budget_per_block[run_op.blocks] = run_budget

            results += [result * blocks_size]
            total_size += blocks_size

        if results:
            result = sum(results) / total_size  # Aggregate RunOp operators
        return result, run_budget_per_block, run_metadata
        

    def run_deterministic(self, run_op, query_id, query, run_metadata):
        # Check for the entry inside the cache
        cache_entry = self.cache.get_entry(query_id, run_op.blocks)

        if not cache_entry:  # Not cached
            # True output never released except in debugging logs
            true_result, blocks_size = self.db.run_query(query, run_op.blocks)

            laplace_scale = run_op.noise_std / np.sqrt(2)
            run_budget = LaplaceCurve(laplace_noise=laplace_scale)
            noise = np.random.laplace(scale=laplace_scale)

        else:  # Cached
            true_result = cache_entry.result

            if run_op.noise_std >= cache_entry.noise_std:
                # We already have a good estimate in the cache
                run_budget = ZeroCurve()
                noise = cache_entry.noise
            else:
                # We need to improve on the cache
                # if not self.variance_reduction:
                # Just compute from scratch and pay for it
                laplace_scale = run_op.noise_std / np.sqrt(2)
                run_budget = LaplaceCurve(laplace_noise=laplace_scale)
                noise = np.random.laplace(scale=laplace_scale)
                # else:
                #     # Var[X] = 2x^2, Y ∼ Lap(y). X might not follow a Laplace distribution!
                #     # Var[aX + bY] = 2(ax)^2 + 2(by)^2 = c
                #     # We set a ∈ [0,1] and b = 1-a
                #     # Then, we maximize y^2 = f(a) = (c - 2(ax)^2)/2(1-a)^2
                #     # We have (1-a)^3 f'(a) = c - 2ax^2
                #     # So we take a = c/(2x^2)
                #     x = cache_entry.noise_std / np.sqrt(2)
                #     c = run_op.noise_std**2
                #     a = c / (2 * (x**2))
                #     b = 1 - a
                #     y = np.sqrt((c - 2 * (a * x) ** 2) / (2 * b**2))

                #     assert np.isclose(2 * (a * x) ** 2 + 2 * (b * y) ** 2, c)

                #     # Get some fresh noise with optimal variance and take a linear combination with the old noise
                #     laplace_scale = y / np.sqrt(2)
                #     fresh_noise = np.random.laplace(scale=laplace_scale)
                #     run_budget = LaplaceCurve(laplace_noise=laplace_scale)
                #     noise = a * noise + b * fresh_noise

        # If we used any fresh noise we need to update the cache
        if not isinstance(run_budget, ZeroCurve):
            cache_entry = CacheEntry(
                result=true_result, noise_std=run_op.noise_std, noise=noise
            )
            self.cache.add_entry(query_id, run_op.blocks, cache_entry)

        result = true_result + noise
        return result, blocks_size, run_budget, run_metadata

    def run_probabilistic(self, run_op, query_id, query, run_metadata):
        pmw = self.cache.get_entry(query_id, run_op.blocks)
        if not pmw:  # If there is no PMW for the blocks then create it
            pmw = self.cache.add_entry(run_op.blocks)

        # True output never released except in debugging logs
        true_result, blocks_size = self.db.run_query(query, run_op.blocks)
        result, run_budget, run_metadata = pmw.run(query, true_result)
        return result, blocks_size, run_budget, run_metadata
