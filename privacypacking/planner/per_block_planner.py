import math
from privacypacking.budget.block import HyperBlock
from privacypacking.cache.cache import A, R
from privacypacking.planner.planner import Planner
from privacypacking.utils.compute_utility_curve import compute_utility_curve
from privacypacking.budget.curves import LaplaceCurve
# TODO: use std instead of pure epsilon


# This planner instance is naive - has only one option for a plan
class PerBlockPlanner(Planner):
    def __init__(
        self, cache, blocks, utility, p, variance_reduction
    ):
        super().__init__(cache)
        self.blocks = blocks
        self.utility = utility
        self.p = p
        # self.max_pure_epsilon = 0.5
        self.variance_reduction = variance_reduction

    def get_execution_plan(self, query_id, block_request):
        """
        For "per-block-planning" a plan has this form: A(R(B1), R(B2), ... , R(Bn))
        """
        n = len(block_request)
        f = compute_utility_curve(self.utility, self.p, n)
        min_pure_epsilon = f[n]
        laplace_scale = 1 / min_pure_epsilon
        noise_std = math.sqrt(2) * laplace_scale

        plan = []
        for x in block_request:
            plan += [R((x, x), noise_std)]
        plan = A(query_id, plan)
        # print(plan)
        cost = self.get_cost(plan)
        if not math.isinf(cost):
            return plan
        return None

    # Simple Cost model
    def get_cost(self, plan):
        query_id = plan.query_id

        for run_op in plan.l:
            block_ids = list(range(run_op.blocks[0], run_op.blocks[-1] + 1))
            hyperblock = HyperBlock({key: self.blocks[key] for key in block_ids})

            cache_entry= self.cache.get_entry(query_id, hyperblock.id)
            if cache_entry is not None:
                # TODO: re-enable variance reduction
                if run_op.noise_std >= cache_entry.noise_std:    # Good enough estimate
                    continue

            # Check if there is enough budget in the hyperblock
            laplace_scale = run_op.noise_std / math.sqrt(2)
            run_budget = LaplaceCurve(laplace_noise=laplace_scale)
            demand = {key: run_budget for key in block_ids}

            if not hyperblock.can_run(demand):
                return math.inf

        return 0
