from privacypacking.budget.block import HyperBlock
from privacypacking.cache.cache import Cache, A, R
from privacypacking.cache.pmw import PMW
import math

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
        pmw = self.get_entry(query_id, hyperblock.id)
        if not pmw:     # If there is no PMW for the hyperblock then create it
            pmw = self.add_entry(hyperblock)
        result, run_budget = pmw.run(query)
        return result, run_budget


    # Cost model    # TODO: remove this functionality from the Cache
    # This is tailored for the per block planning
    def get_cost(self, plan, blocks):   # Cost is either infinite or 0 in this implementation
        if isinstance(plan, A):         # Aggregate cost of arguments/operators
            return sum([self.get_cost(x, blocks) for x in plan.l])

        elif isinstance(plan, R):       # Get cost of Run operator
            block_ids = list(range(plan.blocks[0], plan.blocks[-1] + 1))
            hyperblock = HyperBlock({key: blocks[key] for key in block_ids})
            
            pmw = self.get_entry(plan.query_id, hyperblock.id)
            if pmw and pmw.queries_ran >= pmw.k:
                # A PMW already exists: if the queries_ran >= threshold then we cannot run
                return math.inf
            
            # Creation of a new PMW costs 0: only one block per pmw so the block has all its budget
            # and no queries ran on it yet
            return 0