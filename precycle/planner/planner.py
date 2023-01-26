from precycle.budget import RenyiBudget
from precycle.budget.curves import ZeroCurve, InfinityCurve
import time


class Planner:
    def __init__(self, cache, budget_accountant, config):
        self.cache = cache
        self.budget_accountant = budget_accountant
        self.cache_type = config.cache.type
        self.blocks_metadata = config.blocks_metadata
        self.max_pure_epsilon = 0.5
        # self.variance_reduction = variance_reduction

    def get_execution_plan(sself, query_id, utility, utility_beta, block_request):
        pass

    # Used only by max/min_cuts_planners
    def get_plan_cost_and_set_cache_types(self, plan, query_id, query):
        cost = ZeroCurve()

        if self.cache_type == "DeterministicCache":
            for run_op in plan.l:
                run_op.cache_type = self.cache_type
                run_cost = self.get_run_cost(
                    self.cache.deterministic_cache, run_op, query_id, query
                )
                if isinstance(run_cost, InfinityCurve):
                    return InfinityCurve()
                cost += run_cost
            return cost

        elif self.cache_type == "ProbabilisticCache":
            for run_op in plan.l:
                run_op.cache_type = self.cache_type
                run_cost = self.get_run_cost(
                    self.cache.probabilistic_cache, run_op, query_id, query
                )
                if isinstance(run_cost, InfinityCurve):
                    return InfinityCurve()
                cost += run_cost
            return cost

        elif self.cache_type == "CombinedCache":
            for run_op in plan.l:
                deterministic_cost = self.get_run_cost(
                    self.cache.deterministic_cache, run_op, query_id, query
                )
                probabilistic_cost = self.get_run_cost(
                    self.cache.probabilistic_cache, run_op, query_id, query
                )

                # Comparison is meaningful because we don't have heterogeneous curves - only Laplace
                if RenyiBudget.from_epsilon_list(
                    probabilistic_cost.epsilons
                ) >= RenyiBudget.from_epsilon_list(deterministic_cost.epsilons):
                    run_op.cache_type = "DeterministicCache"
                    run_cost = deterministic_cost
                    # print("Deterministic Won")
                else:
                    run_op.cache_type = "ProbabilisticCache"
                    run_cost = probabilistic_cost
                    # print("Probabilistic Won")

                # time.sleep(2)

                if isinstance(run_cost, InfinityCurve):
                    return InfinityCurve()
                cost += run_cost

        return cost

    def get_run_cost(self, cache, run_op, query_id, query):
        run_budget = cache.estimate_run_budget(
            query_id, query, run_op.blocks, run_op.noise_std
        )
        # Check if there is enough budget in the blocks
        if not self.budget_accountant.can_run(run_op.blocks, run_budget):
            return InfinityCurve()
        return run_budget
