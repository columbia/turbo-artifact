from privacypacking.cache.cache import Cache
from privacypacking.budget.block import HyperBlock
import yaml


class DeterministicCache(Cache):
    def __init__(self,):
        self.key_values = {}

    def add_entry(self, query_id, hyperblock_id, result):
        if query_id not in self.key_values:
            self.key_values[query_id] = {}
        if hyperblock_id not in self.key_values[query_id]:
            self.key_values[query_id].update({hyperblock_id: result})

    def get_entry(self, query_id, hyperblock_id):
        result = None
        if query_id in self.key_values:
            if hyperblock_id in self.key_values[query_id]:
                result = self.key_values[query_id][hyperblock_id]
        return result

    def run(self, query_id, query, run_budget, hyperblock: HyperBlock):
        result = self.get_entry(query_id, hyperblock.id)
        
        if not result:  # If result is not in the cache run fresh and store
            result = hyperblock.run_dp(query, run_budget)
            self.add_entry(query_id, hyperblock.id, result)  # Add result in cache

        return result, run_budget

    def dump(
        self,
    ):
        res = yaml.dump(self.key_values)
        print("Results", res)
