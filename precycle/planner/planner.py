class Planner:
    def __init__(self, cache, budget_accountant, enable_caching, enable_dp, cache_type):
        self.cache = cache
        self.cache_type = cache_type
        self.budget_accountant = budget_accountant
        # self.variance_reduction = variance_reduction
        self.enable_caching = enable_caching
        self.enable_dp = enable_dp
        self.max_pure_epsilon = 0.5

    def get_execution_plan(sself, query_id, utility, utility_beta, block_request):
        pass
