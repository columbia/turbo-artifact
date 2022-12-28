class Planner:
    def __init__(self, cache, blocks, planner_args):
        self.cache = cache
        self.blocks = blocks
        self.variance_reduction = planner_args.variance_reduction
        self.enable_caching = planner_args.enable_caching
        self.enable_dp = planner_args.enable_dp
        self.max_pure_epsilon = 0.5

    def get_execution_plan(sself, query_id, utility, utility_beta, block_request):
        pass
