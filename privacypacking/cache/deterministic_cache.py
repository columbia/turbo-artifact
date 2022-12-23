# from privacypacking.budget.utils import from_pure_epsilon_to_budget
import numpy as np
import yaml

from privacypacking.budget.block import HyperBlock
from privacypacking.budget.curves import LaplaceCurve, ZeroCurve
from privacypacking.cache.cache import Cache


class DeterministicCache(Cache):
    def __init__(self, variance_reduction):
        self.key_values = {}
        self.variance_reduction = variance_reduction

    def add_entry(self, query_id, hyperblock_id, result, budget, noise):
        if query_id not in self.key_values:
            self.key_values[query_id] = {}
        self.key_values[query_id].update({hyperblock_id: (result, budget, noise)})

    def get_entry(self, query_id, hyperblock_id):
        # The cache shouldn't have to care about how we do DP accounting (RDP, Pure DP, etc).
        # It only cares about the true result, the actual noise, and the distribution from which we sampled the noise
        # When necessary we can map the std to an epsilon
        if query_id in self.key_values:
            if hyperblock_id in self.key_values[query_id]:
                (result, budget, noise) = self.key_values[query_id][hyperblock_id]
                return result, budget, noise
        return None, None, None

    def run(self, query_id, query, noise_std, hyperblock: HyperBlock):
        """
        noise_std is the std of the noise that a sensitivity 1 query is willing to accept
        laplace_scale = 1/pure_epsilon and std = \sqrt{2} * laplace_scale
        For queries that have sensitivity != 1 we need to multiply the noise, but for now we just have counts.
        """
        # TODO: maybe use alpha, beta, or std instead of noise scale? And add sensitivity argument?
        run_budget = None

        true_result, cached_noise_std, cached_noise = self.get_entry(
            query_id, hyperblock.id
        )
        if true_result is None:  # Not cached ever
            true_result = hyperblock.run(query)  # Run without noise
            cached_noise_std = 0
            cached_noise = 0

        if noise_std > cached_noise_std:
            # We already have a good estimate in the cache
            run_budget = ZeroCurve()
            noise = cached_noise
        else:
            # We need to improve on the (potentially empty) cache
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
                x = cached_noise_std / np.sqrt(2)
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

                # Store this improved result
                self.add_entry(query_id, hyperblock.id, true_result, noise_std, noise)

        result = true_result + noise
        return result, run_budget

    def run_with_budget(self, query_id, query, demand_budget, hyperblock: HyperBlock):
        run_budget = None

        true_result, cached_budget, cached_noise = self.get_entry(
            query_id, hyperblock.id
        )
        if true_result is None:  # Not cached ever
            true_result = hyperblock.run(query)  # Run without noise
            run_budget = demand_budget
            noise = run_budget.compute_noise()
        else:  # Cached already with some budget and noise
            if (
                demand_budget.pure_epsilon <= cached_budget.pure_epsilon
            ):  # If cached budget is enough
                noise = cached_noise
            else:  # If cached budget is not enough
                if self.variance_reduction:  # If optimization is enabled
                    run_budget = from_pure_epsilon_to_budget(
                        demand_budget.pure_epsilon - cached_budget.pure_epsilon
                    )
                    run_noise = run_budget.compute_noise()
                    noise = (
                        cached_budget.pure_epsilon * cached_noise
                        + run_budget.pure_epsilon * run_noise
                    ) / (cached_budget.pure_epsilon + run_budget.pure_epsilon)
                else:  # If optimization is not enabled
                    run_budget = demand_budget
                    noise = run_budget.compute_noise()

        result = true_result + noise

        if run_budget is not None:
            self.add_entry(query_id, hyperblock.id, true_result, demand_budget, noise)
        return result, run_budget

    def dump(self):
        res = yaml.dump(self.key_values)
        print("Results", res)
