class Planner:
    def __init__(self, cache, budget_accountant, config):
        self.cache = cache
        self.budget_accountant = budget_accountant
        self.cache_type = config.cache.type
        self.blocks_metadata = config.blocks_metadata
        self.max_pure_epsilon = 0.5
        self.config = config
        self.probabilistic_cfg = self.config.cache.probabilistic_cfg

    def get_execution_plan(sself, task):
        pass
