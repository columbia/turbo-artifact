from precycle.cache.pmw import PMW
from precycle.cache.cache import Cache


class ProbabilisticCache(Cache):
    def __init__(self, config):
        raise NotImplementedError


class MockProbabilisticCache(Cache):
    def __init__(self, config):
        self.key_values = {}
        self.config = config
        self.pmw_args = config.cache.pmw_cfg

    def add_entry(self, blocks):
        pmw = PMW(blocks, self.config.blocks_metadata, **self.pmw_args)
        self.key_values[blocks] = pmw
        return pmw

    def get_entry(self, query_id, blocks):
        if blocks in self.key_values:
            return self.key_values[blocks]
        return None

    def estimate_run_budget(self, query_id, blocks, noise_std):
        """
        Checks the cache and returns the budget we need to spend if we want to run this query with given accuracy guarantees.
        """

        pmw = self.get_entry(query_id, blocks)
        run_budget = (
            PMW(
                blocks, self.config.blocks_metadata, **self.pmw_args
            ).estimate_run_budget()
            if not pmw
            else pmw.estimate_run_budget()
        )
        # TODO: This is leaking privacy, assume we have a good estimate already.
        return run_budget
