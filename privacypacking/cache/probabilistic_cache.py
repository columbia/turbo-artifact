import math

from privacypacking.budget.block import Block, HyperBlock
from privacypacking.cache.cache import A, Cache, R
from privacypacking.cache.pmw import PMW


class ProbabilisticCache(Cache):
    def __init__(self, **pmw_args):
        self.key_values = {}
        self.pmw_args = pmw_args

    def add_entry(self, hyperblock: HyperBlock):
        pmw = PMW(hyperblock, **self.pmw_args)
        self.key_values[hyperblock.id] = pmw
        return pmw

    def get_entry(self, query_id, hyperblock_id):
        if hyperblock_id in self.key_values:
            return self.key_values[hyperblock_id]
        return None

    def run(self, query_id, query, run_budget, hyperblock: HyperBlock):
        pmw = self.get_entry(query_id, hyperblock.id)
        if not pmw:  # If there is no PMW for the hyperblock then create it
            pmw = self.add_entry(hyperblock)
        result, run_budget = pmw.run(query)
        return result, run_budget, 0  # Not logging noise

    # Cost model    # TODO: remove this functionality from the Cache
    # This is tailored for the per block planning
    def get_cost(
        self, plan, blocks
    ):  # Cost is either infinite or 0 in this implementation
        if isinstance(plan, A):  # Aggregate cost of arguments/operators
            return sum([self.get_cost(x, blocks) for x in plan.l])

        elif isinstance(plan, R):  # Get cost of Run operator
            block_ids = list(range(plan.blocks[0], plan.blocks[-1] + 1))
            hyperblock = HyperBlock({key: blocks[key] for key in block_ids})

            # NOTE: This is different from the deterministic cache. Any run might cost budget,
            # whether the cache is empty or not, and whether we hit or not.
            pmw = self.get_entry(plan.query_id, hyperblock.id)
            if pmw is None:
                # Use the defaults or the same config as `add_entry`
                budget = PMW.worst_case_cost(**self.pmw_args)
            else:
                budget = PMW.worst_case_cost(
                    pmw.nu, pmw.ro, pmw.standard_svt, pmw.local_svt_max_queries
                )

            demand = {key: budget for key in block_ids}
            if not hyperblock.can_run(demand):
                return math.inf  # This hyperblock does not have enough budget
            return 0
