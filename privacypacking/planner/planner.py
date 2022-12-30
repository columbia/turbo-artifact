class Planner:
    def __init__(self, cache, blocks, variance_reduction, enable_caching, enable_dp):
        self.cache = cache
        self.blocks = blocks
        self.variance_reduction = variance_reduction
        self.enable_caching = enable_caching
        self.enable_dp = enable_dp
        self.max_pure_epsilon = 0.5

    def get_execution_plan(sself, query_id, utility, utility_beta, block_request):
        pass
