from privacypacking.cache.cache import Cache, R, A
from privacypacking.budget.block import HyperBlock
import yaml
import math
import numpy as np
from privacypacking.budget import (
    ALPHAS,
    BasicBudget,
    Budget,
    RenyiBudget,
    SparseHistogram,
)
from privacypacking.budget.curves import GaussianCurve


class DeterministicCache(Cache):
    def __init__(self, variance_reduction):
        self.key_values = {}
        self.variance_reduction = variance_reduction

    def add_entry(self, query_id, hyperblock_id, result, budget, noise):
        if query_id not in self.key_values:
            self.key_values[query_id] = {}
        self.key_values[query_id].update({hyperblock_id: (result, budget, noise)})

    def get_entry(self, query_id, hyperblock_id):
        if query_id in self.key_values:
            if hyperblock_id in self.key_values[query_id]:
                (result, budget, noise) = self.key_values[query_id][hyperblock_id]
                return result, budget, noise
        return None, None, None

    def run(self, query_id, query, demand_budget, hyperblock: HyperBlock):
        run_budget = None

        true_result, cached_budget, cached_noise = self.get_entry(
            query_id, hyperblock.id
        )
        if true_result is None:  # Not cached ever
            true_result = hyperblock.run(query)  # Run without noise
            run_budget = demand_budget
            noise = self.compute_noise(run_budget)
        else:  # Cached already with some budget and noise
            if (
                demand_budget.epsilon <= cached_budget.epsilon
            ):  # If cached budget is enough
                noise = cached_noise
            else:  # If cached budget is not enough
                if self.variance_reduction:  # If optimization is enabled
                    run_budget = demand_budget - cached_budget
                    run_noise = self.compute_noise(run_budget)
                    noise = (
                        cached_budget.epsilon * cached_noise
                        + run_budget.epsilon * run_noise
                    ) / (cached_budget + run_budget).epsilon
                else:  # If optimization is not enabled
                    run_budget = demand_budget
                    noise = self.compute_noise(run_budget)

        result = true_result + noise

        if run_budget is not None:
            self.add_entry(query_id, hyperblock.id, true_result, demand_budget, noise)
        return result, run_budget

    def compute_noise(self, budget):  # TODO: move this elsewhere
        sensitivity = 1
        if isinstance(budget, BasicBudget):
            noise = np.random.laplace(scale=sensitivity / budget.epsilon)
        # elif isinstance(budget, GaussianCurve):
        # noise = np.random.normal(scale=sensitivity * budget.sigma)
        # elif isinstance(budget, RenyiBudget):
        # raise NotImplementedError("Try to find the best sigma?")
        return noise

    # Cost model for minimizing budget
    def get_entry_budget(self, query_id, blocks):
        result, budget, _ = self.get_entry(query_id, blocks)
        if result is not None:
            return budget.epsilon
        return 0.0

    def dump(self):
        res = yaml.dump(self.key_values)
        print("Results", res)
