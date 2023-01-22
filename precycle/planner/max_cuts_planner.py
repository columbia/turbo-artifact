import math
from precycle.executor import A, R
from precycle.planner.planner import Planner
from precycle.utils.compute_utility_curve import compute_utility_curve
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

            print("\n++++++++++++++++++++++++++++++++++++++++++++")
            print("min pure epsilon", min_pure_epsilon)
            print("total size", blocks_size)
            print("sensitivity", sensitivity)
            print("laplace scale", laplace_scale)
            print("noise std", noise_std)
            print("++++++++++++++++++++++++++++++++++++++++++++\n")

            run_ops += [
                R(blocks=(b, b), noise_std=noise_std, cache_type=self.cache_type)
            ]

        plan = A(run_ops)

        cost = self.get_cost(plan, task.query_id)
        if not math.isinf(cost):
            plan.cost = cost
            return plan
        return None

    # TODO: Move this elsewhere
    # Simple Cost model - returns 0 or inf. - used only by max/min_cuts_planners
    def get_cost(self, plan, query_id):
        for run_op in plan.l:
            run_budget = self.cache.estimate_run_budget(
                query_id, run_op.blocks, run_op.noise_std
            )
        # Check if there is enough budget in the blocks
        if not self.budget_accountant.can_run(run_op.blocks, run_budget):
            return math.inf
        return 0
