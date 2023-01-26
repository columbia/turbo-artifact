import math
from precycle.executor import A, R
from precycle.planner.planner import Planner
from precycle.utils.compute_utility_curve import compute_utility_curve
from precycle.budget.curves import InfinityCurve
from precycle.utils.utils import get_blocks_size


class MinCutsPlanner(Planner):
    def __init__(self, cache, budget_accountant, config):
        super().__init__(cache, budget_accountant, config)

    def get_execution_plan(self, task):

        """For "MinCutsPlanner" a plan has this form: A(R(B1,B2, ... , Bn))"""

        # 0 Aggregations
        blocks_size = get_blocks_size(task.blocks, self.blocks_metadata)
        min_pure_epsilon = compute_utility_curve(
            task.utility, task.utility_beta, blocks_size, 1
        )
        sensitivity = 1 / blocks_size
        laplace_scale = sensitivity / min_pure_epsilon
        noise_std = math.sqrt(2) * laplace_scale

        plan = A(l=[R(blocks=task.blocks, noise_std=noise_std)])
        # Gets the cost of the plan and sets the cache type of each run operator

        cost = self.get_plan_cost_and_set_cache_types(plan, task.query_id, task.query)

        if not isinstance(cost, InfinityCurve):
            plan.cost = cost
            return plan
        return None

        # print("\n++++++++++++++++++++++++++++++++++++++++++++")
        # print("min pure epsilon", min_pure_epsilon)
        # print("total size", blocks_size)
        # print("sensitivity", sensitivity)
        # print("laplace scale", laplace_scale)
        # print("noise std", noise_std)
        # print("++++++++++++++++++++++++++++++++++++++++++++\n")
