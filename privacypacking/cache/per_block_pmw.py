from privacypacking.budget.block import Block
from privacypacking.cache.cache import A, C, Cache
from privacypacking.cache.pmw import PMW
from privacypacking.cache.utils import get_splits


class PerBlockPMW(Cache):
    def __init__(self):
        self.all_PMW = {}  # One PMW per block

    def addPMW(self, block: Block):
        pmw = PMW(block)
        self.all_PMW[block.id] = pmw
        return pmw

    def getPMW(self, block_id):
        if block_id in self.all_PMW:
            return self.all_PMW[block_id]
        return None

    def run_cache(self, query_tensor, block: Block):
        pmw = self.getPMW(block.id)
        # If there is no PMW for the block then create it (creation happens on demand not eagerly)
        if pmw is None:
            pmw = self.addPMW(block)

        result, run_budget = pmw.run_cache(query_tensor)
        return result, run_budget

    def run_cache_outdated(self, query_id, block, budget):
        pmw = self.getPMW(block)
        # If there is no PMW for the block then create it (creation happens on demand not eagerly)
        if pmw is None:
            pmw = self.addPMW(block)
        result, run_budget = pmw.run_cache(query_id, block, budget)
        return result, run_budget

    def get_execution_plan(self, query_id, blocks, budget):
        """
        For per-block-pmu all plans have this form: A(C(B1), C(B2), ... , C(Bn))
        """
        num_aggregations = len(blocks) - 1
        plan = []
        splits = get_splits(blocks, num_aggregations)
        for split in splits:
            # print("split", split)
            for x in split:
                x = (x[0], x[-1])
                plan += [C(query_id, x, budget)]

            # Only one split for per-block-pmu - we stop here
            if len(plan) == 1:
                return plan[0]
            else:
                return A(plan)
        return None
