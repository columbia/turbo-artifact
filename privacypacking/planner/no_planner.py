import math

from privacypacking.budget.block import HyperBlock
from privacypacking.cache.cache import A, R
from privacypacking.planner.planner import Planner
from privacypacking.utils.compute_utility_curve import compute_utility_curve


class NoPlanner(Planner):
    def __init__(
        self, cache, blocks, utility, optimization_objective, variance_reduction
    ):
        super().__init__(cache)
        self.blocks = blocks
        self.utility = utility
        self.p = 0.00001  # Probability that accuracy won't be respected
        # self.max_pure_epsilon = 0.5
        self.variance_reduction = variance_reduction

    def get_execution_plan(self, query_id, block_request, budget):

        n = len(block_request)
        f = compute_utility_curve(self.utility, self.p, n)
        min_pure_epsilon = f[n]

        bs_tuple = (block_request[0], block_request[-1])
        plan = A([R(query_id=query_id, blocks=bs_tuple, budget=budget)])
        cost = self.get_cost(plan)
        # Get the cost of the plan - if infinite it's not eligible
        if not math.isinf(cost):
            return plan
        return None

    # Simple Cost model
    def get_cost(self, plan):
        if isinstance(plan, A):  # Aggregate cost of arguments/operators
            return sum([self.get_cost(x) for x in plan.l])
        elif isinstance(plan, R):  # Get cost of Run operator
            block_ids = list(range(plan.blocks[0], plan.blocks[-1] + 1))
            hyperblock = HyperBlock({key: self.blocks[key] for key in block_ids})

            # TODO: use std instead
            result, cached_budget, _ = self.cache.get_entry(
                plan.query_id, hyperblock.id
            )
            if result is not None:
                demand_pure_epsilon = max(
                    plan.budget.pure_epsilon - cached_budget.pure_epsilon, 0
                )
                if not self.variance_reduction:
                    demand_pure_epsilon = plan.budget.pure_epsilon
                demand_budget = from_pure_epsilon_to_budget(demand_pure_epsilon)
                demand = {key: demand_budget for key in block_ids}
            else:
                demand = {key: plan.budget for key in block_ids}
            if not hyperblock.can_run(demand):
                return math.inf  # This hyperblock does not have enough budget
            return 1  # Even if there is at least a little budget left in the hyperblock we assume the cost is 1
