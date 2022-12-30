import math
from privacypacking.cache.cache import A, R
from privacypacking.planner.planner import Planner
from privacypacking.budget.block import HyperBlock
from privacypacking.budget.curves import LaplaceCurve
from privacypacking.utils.compute_utility_curve import compute_utility_curve


class MinCutsPlanner(Planner):
    def __init__(self, cache, blocks, planner_args):
        super().__init__(cache, blocks, **planner_args)

    def get_execution_plan(self, query_id, utility, utility_beta, block_request):
        """
        For "MinCutsPlanner" a plan has this form: A(R(B1,B2, ... , Bn))
        """
        # 0 Aggregations
        min_pure_epsilon = compute_utility_curve(utility, utility_beta, 1)
        laplace_scale = 1 / min_pure_epsilon
        noise_std = math.sqrt(2) * laplace_scale

        bs_tuple = (block_request[0], block_request[-1])
        plan = A(query_id=query_id, l=[R(bs_tuple, noise_std)])
        
        cost = 0
        if self.enable_dp:
            cost = self.get_cost(plan)
        if not math.isinf(cost):
            plan.cost = cost
            return plan
        return None

    # Simple Cost model     # TODO: Move this elsewhere
    def get_cost(self, plan):
        query_id = plan.query_id

        for run_op in plan.l:
            block_ids = list(range(run_op.blocks[0], run_op.blocks[-1] + 1))
            hyperblock = HyperBlock({key: self.blocks[key] for key in block_ids})

            if self.enable_caching:
                cache_entry = self.cache.get_entry(query_id, hyperblock.id)
                if cache_entry is not None:
                    # TODO: re-enable variance reduction
                    if (
                        run_op.noise_std >= cache_entry.noise_std
                    ):  # Good enough estimate
                        continue

            # Check if there is enough budget in the hyperblock
            laplace_scale = run_op.noise_std / math.sqrt(2)
            run_budget = LaplaceCurve(laplace_noise=laplace_scale)
            demand = {key: run_budget for key in block_ids}

            if not hyperblock.can_run(demand):
                return math.inf

        return 0
