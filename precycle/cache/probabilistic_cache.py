from typing import Dict
from cache.cache import Cache
from cache.pmw import PMW


class ProbabilisticCache(Cache):
    def __init__(self, cache_cfg: Dict = {}):
        self.key_values = {}
        self.pmw_args = cache_cfg

    def add_entry(self, hyperblock):
        pmw = PMW(hyperblock, **self.pmw_args)
        self.key_values[hyperblock.id] = pmw
        return pmw

    def get_entry(self, query_id, hyperblock_id):
        if hyperblock_id in self.key_values:
            return self.key_values[hyperblock_id]
        return None

    def run(self, query_id, query, run_budget, hyperblock):
        pmw = self.get_entry(query_id, hyperblock.id)
        if not pmw:  # If there is no PMW for the hyperblock then create it
            pmw = self.add_entry(hyperblock)
        result, run_budget, run_metadata = pmw.run(query)
        return result, run_budget, run_metadata

    def estimate_run_budget(self, query_id, hyperblock, noise_std):
        """
        Checks the cache and returns the budget we need to spend if we want to run this query with given accuracy guarantees.
        """

        # NOTE: This is different from the deterministic cache. Any run might cost budget,
        # whether the cache is empty or not, and whether we hit or not.
        pmw = self.get_entry(query_id, hyperblock.id)
        if pmw is None:
            # Use the defaults or the same config as `add_entry`
            # TODO: This is leaking privacy, assume we have a good estimate already.
            run_budget = PMW.estimate_run_budget(n=hyperblock.size, **self.pmw_args)
        else:
            run_budget = PMW.estimate_run_budget(
                n=hyperblock.size,
                nu=pmw.nu,
                ro=pmw.ro,
                standard_svt=pmw.standard_svt,
                local_svt_max_queries=pmw.local_svt_max_queries,
            )

        return run_budget
