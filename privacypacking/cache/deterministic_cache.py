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
    def __init__(
        self, variance_reduction
    ):
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
        if true_result is None:                     # Not cached ever
            true_result = hyperblock.run(query)     # Run without noise
            run_budget = demand_budget
            noise = self.compute_noise(run_budget)
        else:                                       # Cached already with some budget and noise
            if demand_budget.epsilon <= cached_budget.epsilon:  # If cached budget is enough
                noise = cached_noise
            else:                                   # If cached budget is not enough
                if self.variance_reduction:         # If optimization is enabled
                    run_budget = demand_budget - cached_budget
                    run_noise = self.compute_noise(run_budget)
                    noise = (cached_budget.epsilon * cached_noise + run_budget.epsilon * run_noise) / \
                            (cached_budget + run_budget).epsilon
                else:                               # If optimization is not enabled
                    run_budget = demand_budget
                    noise = self.compute_noise(run_budget)

        result = true_result + noise

        if run_budget is not None:
            self.add_entry(query_id, hyperblock.id, true_result, demand_budget, noise)
        return result, run_budget

    def compute_noise(self, budget):    #TODO: move this elsewhere
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




    # # Minimizing aggregations
    # # Cost model    # TODO: remove this functionality from the Cache
    # def get_cost(self, plan, blocks, structure_constraint=False, branching_factor=2):
    #     if isinstance(plan, A):  # Aggregate cost of arguments/operators
    #         return sum([self.get_cost(x, blocks) for x in plan.l])

    #     elif isinstance(plan, R):  # Get cost of Run operator
    #         # print(f"getting cost for {plan}")

    #         if structure_constraint and not self.satisfies_constraint(
    #             plan.blocks, branching_factor
    #         ):
    #             # print("Cost: Not binary!")
    #             return math.inf

    #         block_ids = list(range(plan.blocks[0], plan.blocks[-1] + 1))
    #         hyperblock = HyperBlock({key: blocks[key] for key in block_ids})

    #         result, _ = self.get_entry_with_budget(
    #             plan.query_id, hyperblock.id, plan.budget
    #         )
    #         if result:
    #             return 1  # Already cached
    #         else:
    #             demand = {key: plan.budget for key in block_ids}
    #             if not hyperblock.can_run(demand):
    #                 # print("Cost: Not enough budget!")
    #                 return math.inf  # This hyperblock does not have enough budget

    #             return 1  # Even if there is at least a little budget left in the hyperblock we assume the cost is 1

    # def satisfies_constraint(self, blocks, branching_factor):
    #     size = blocks[1] - blocks[0] + 1
    #     if not math.log(size, branching_factor).is_integer():
    #         return False
    #     if (blocks[0] % size) != 0:
    #         return False
    #     return True
