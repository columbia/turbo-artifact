class Planner:
    def __init__(self, cache, budget_accountant, config):
        self.cache = cache
        self.config = config
        self.cache_type = config.cache.type
        self.budget_accountant = budget_accountant
        self.blocks_metadata = config.blocks_metadata
        self.probabilistic_cfg = self.config.cache.probabilistic_cfg

    def get_execution_plan(sself, task):
        pass
