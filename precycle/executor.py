import numpy as np
from loguru import logger
from typing import Dict, Tuple
from collections import namedtuple
from precycle.budget.curves import LaplaceCurve, ZeroCurve
from precycle.cache.deterministic_cache import CacheEntry
from precycle.utils.utils import get_blocks_size


class RDet:
    def __init__(self, blocks, noise_std) -> None:
        self.blocks = blocks
        self.noise_std = noise_std

    def __str__(self):
        return f"RunDet({self.blocks}, {self.noise_std})"


class RProb:
    def __init__(self, blocks, alpha, beta) -> None:
        self.blocks = blocks
        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        return f"RunProb({self.blocks}, {self.alpha}, {self.beta})"


class A:
    def __init__(self, l, cost=None) -> None:
        self.l = l
        self.cost = cost

    def __str__(self):
        return f"Aggregate({[str(l) for l in self.l]})"


RunReturnValue = namedtuple(
    "RunReturnValue",
    [
        "noisy_result",
        "true_result",
        "run_budget",
        "run_metadata",
        "noise_std",
        "noise",
    ],
)


class Executor:
    def __init__(self, cache, db, config) -> None:
        self.db = db
        self.cache = cache
        self.config = config

    def execute_plan(self, plan, task) -> Tuple[float, Dict]:
        """
        run_budget: the budget that will be consumed from the blocks after running the query
        """
        result = None
        total_size = 0
        run_ops_metadata = {}
        run_budget_per_block = {}

        results = []
        for run_op in plan.l:
            blocks_size = get_blocks_size(run_op.blocks, self.config.blocks_metadata)

            if isinstance(run_op, RDet):
                run_return_value = self.run_deterministic(
                    run_op, task.query_id, task.query
                )
                # Use the result to update the Probabilistic cache as well
                if self.config.cache.type == "CombinedCache":
                    self.cache.probabilistic_cache.update_entry(
                        task.query,
                        run_op.blocks,
                        run_return_value.true_result,
                        run_op.noise_std,
                        run_return_value.noise,
                    )
            elif isinstance(run_op, RProb):
                run_return_value = self.run_probabilistic(run_op, task.query)
                # Use the result to update the Deterministic cache as well
                if (
                    self.config.cache.type == "CombinedCache"
                    and run_return_value.run_metadata["hard_query"]
                ):
                    self.cache.deterministic_cache.update_entry(
                        task.query_id,
                        run_op.blocks,
                        run_return_value.true_result,
                        run_return_value.noise_std,
                        run_return_value.noise,
                    )

            run_ops_metadata[f"R({run_op.blocks})"] = run_return_value.run_metadata
            run_budget_per_block[run_op.blocks] = run_return_value.run_budget

            results += [run_return_value.noisy_result * blocks_size]
            total_size += blocks_size

        if results:
            result = sum(results) / total_size  # Aggregate RunOp operators
        return result, run_budget_per_block, run_ops_metadata

    def run_deterministic(self, run_op, query_id, query):
        run_op_metadata = {}
        run_op_metadata["cache_type"] = "DeterministicCache"

        blocks_size = get_blocks_size(run_op.blocks, self.config.blocks_metadata)
        sensitivity = 1 / blocks_size

        # Check for the entry inside the cache
        cache_entry = self.cache.deterministic_cache.get_entry(query_id, run_op.blocks)

        if not cache_entry:  # Not cached
            run_op_metadata["hard_query"] = True
            # True output never released except in debugging logs
            true_result = self.db.run_query(query, run_op.blocks)
            laplace_scale = run_op.noise_std / np.sqrt(2)
            run_budget = LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
            noise = np.random.laplace(scale=laplace_scale)

        else:  # Cached
            run_op_metadata["hard_query"] = False
            true_result = cache_entry.result

            if run_op.noise_std >= cache_entry.noise_std:
                # We already have a good estimate in the cache
                run_budget = ZeroCurve()
                noise = cache_entry.noise
            else:
                variance_reduction = False  # No VR for now

                # We need to improve on the cache
                if not variance_reduction:
                    # Just compute from scratch and pay for it
                    laplace_scale = run_op.noise_std / np.sqrt(2)
                    run_budget = LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
                    noise = np.random.laplace(scale=laplace_scale)
                # else:
                #     # TODO: re-enable variance reduction
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
                #     run_budget = LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
                #     noise = a * cache_entry.noise + b * fresh_noise

        # If we used any fresh noise we need to update the cache
        if not isinstance(run_budget, ZeroCurve):
            cache_entry = CacheEntry(
                result=true_result, noise_std=run_op.noise_std, noise=noise
            )
            self.cache.deterministic_cache.add_entry(
                query_id, run_op.blocks, cache_entry
            )

        noisy_result = true_result + noise
        rv = RunReturnValue(
            noisy_result,
            true_result,
            run_budget,
            run_op_metadata,
            run_op.noise_std,
            noise,
        )
        return rv

    def run_probabilistic(self, run_op, query):
        pmw = self.cache.probabilistic_cache.get_entry(run_op.blocks)
        obj = pmw if pmw else self.config.cache.pmw_accuracy

        if run_op.alpha > obj.alpha or run_op.beta < obj.beta:
            pmw = self.cache.probabilistic_cache.add_entry(
                run_op.blocks, run_op.alpha, run_op.beta, pmw
            )
            logger.error(
                "Plan requires more powerful PMW than the one cached. We decided this wouldn't happen."
            )
        elif not pmw:
            pmw = self.cache.probabilistic_cache.add_entry(
                run_op.blocks, obj.alpha, obj.beta
            )

        # True output never released except in debugging logs
        true_result = self.db.run_query(query, run_op.blocks)
        noisy_result, run_budget, run_op_metadata = pmw.run(query, true_result)
        run_op_metadata["cache_type"] = "ProbabilisticCache"
        noise = noisy_result - true_result
        rv = RunReturnValue(
            noisy_result, true_result, run_budget, run_op_metadata, pmw.noise_std, noise
        )
        return rv
