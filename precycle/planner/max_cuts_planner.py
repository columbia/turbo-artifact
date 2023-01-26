import math
from precycle.executor import A, R
from precycle.planner.planner import Planner
from precycle.utils.compute_utility_curve import compute_utility_curve
from precycle.budget.curves import InfinityCurve
from precycle.utils.utils import get_blocks_size


class MaxCutsPlanner(Planner):
    def __init__(self, cache, budget_accountant, config):
        super().__init__(cache, budget_accountant, config)

    def get_execution_plan(self, task):

        """For "MaxCutsPlanner" a plan has this form: A(R(B1), R(B2), ... , R(Bn))"""

        # Num-blocks Aggregations
        block_request = range(task.blocks[0], task.blocks[1] + 1)
        num_blocks = len(block_request)
        blocks_size = get_blocks_size(task.blocks, self.blocks_metadata)

        min_pure_epsilon = compute_utility_curve(
            task.utility, task.utility_beta, blocks_size, num_blocks
        )

        run_ops = []
        for b in block_request:
            block_size = get_blocks_size(b, self.blocks_metadata)
            sensitivity = 1 / block_size
            laplace_scale = sensitivity / min_pure_epsilon
            noise_std = math.sqrt(2) * laplace_scale

            run_ops += [R(blocks=(b, b), noise_std=noise_std)]

        plan = A(run_ops)
        # Gets the cost of the plan and sets the cache type of each run operator
        cost = self.get_plan_cost_and_set_cache_types(plan, task.query_id, task.query)

        if not isinstance(cost, InfinityCurve):
            plan.cost = cost
            return plan
        return None
