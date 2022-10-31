from privacypacking.budget.block import HyperBlock
from privacypacking.cache.cache import Cache
from privacypacking.cache.pmw import PMW


class ProbabilicticCache(Cache):
    def __init__(self):
        self.key_values = {}

    def add_entry(self, hyperblock: HyperBlock):
        pmw = PMW(hyperblock)
        self.key_values[hyperblock.id] = pmw
        return pmw

    def get_entry(self, query_id, hyperblock_id):
        if hyperblock_id in self.key_values:
            return self.key_values[hyperblock_id]
        return None

    def run(self, query_id, query, run_budget, hyperblock: HyperBlock):
        pmw = self.get_entry(hyperblock.id)
        # If there is no PMW for the hyperblock then create it (creation happens on demand not eagerly)
        if pmw is None:
            pmw = self.add_entry(hyperblock)

        result, run_budget = pmw.run(query)
        return result, run_budget