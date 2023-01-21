import math
from precycle.executor import A, R
from precycle.planner.planner import Planner
from precycle.budget.curves import LaplaceCurve
from precycle.utils.compute_utility_curve import compute_utility_curve


class MaxCutsPlanner(Planner):
    def __init__(self, cache, budget_accountant, planner_args):
        assert planner_args.get("enable_caching") == True
        assert planner_args.get("enable_dp") == True
        super().__init__(cache, budget_accountant, **planner_args)

    def get_execution_plan(self, task):

        """For "MaxCutsPlanner" a plan has this form: A(R(B1), R(B2), ... , R(Bn))"""

        # Num-blocks Aggregations
        block_request = range(task.blocks[0], task.blocks[1] + 1)
        num_blocks = len(block_request)
        total_data_size = num_blocks * self.blocks_metadata["block_size"]
        min_pure_epsilon = compute_utility_curve(
            task.utility, task.utility_beta, total_data_size, num_blocks
        )
        laplace_scale = 1 / min_pure_epsilon
        noise_std = math.sqrt(2) * laplace_scale

        plan = A(
            l=[
                R(blocks=(x, x), noise_std=noise_std, cache_type=self.cache_type)
                for x in block_request
            ]
        )

        cost = self.get_cost(plan, task.query_id)
        if not math.isinf(cost):
            plan.cost = cost
            return plan
        return None

    # TODO: Move this elsewhere
    # Simple Cost model - returns 0 or inf. - used only by max/min_cuts_planners
    def get_cost(self, plan, query_id):
        for run_op in plan.l:
            if self.enable_caching:
                run_budget = self.cache.estimate_run_budget(
                    query_id, run_op.blocks, run_op.noise_std
                )
            else:
                laplace_scale = run_op.noise_std / math.sqrt(2)
                run_budget = LaplaceCurve(laplace_noise=laplace_scale)

            # Check if there is enough budget in the blocks
            if not self.budget_accountant.can_run(run_op.blocks, run_budget):
                return math.inf
        return 0
