import math
from precycle.executor import A, R
from precycle.planner.planner import Planner
from precycle.budget.curves import LaplaceCurve
from precycle.utils.compute_utility_curve import compute_utility_curve


class MinCutsPlanner(Planner):
    def __init__(self, cache, budget_accountant, planner_args):
        super().__init__(cache, budget_accountant, **planner_args)

    def get_execution_plan(self, task):

        """For "MinCutsPlanner" a plan has this form: A(R(B1,B2, ... , Bn))"""

        # 0 Aggregations
        min_pure_epsilon = compute_utility_curve(task.utility, task.utility_beta, 1)
        laplace_scale = 1 / min_pure_epsilon
        noise_std = math.sqrt(2) * laplace_scale

        plan = A(
            l=[R(blocks=task.blocks, noise_std=noise_std, cache_type=self.cache_type)]
        )

        cost = 0
        if self.enable_dp:
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

            # Check if there is enough budget in the hyperblock
            if not self.budget_accountant.can_run(run_op.blocks, run_budget):
                return math.inf
        return 0
