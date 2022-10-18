from privacypacking.cache.utils import get_splits
from privacypacking.cache.pmw import PMW
from privacypacking.cache.cache import Cache, A, C

class PerBlockPMW(Cache):
    def __init__(self, scheduler):
        self.scheduler = scheduler      # todo: find a way to remove this
        self.all_PMW = {}               # One PMW per block
    
    def addPMW(self, block):
        pmw = PMW(self.scheduler, block)
        self.all_PMW[block] = pmw
        return pmw

    def getPMW(self, block):
        return self.all_PMW[block] or None

    def run_cache(self, query_id, block, budget):
        pmw = self.getPMW(block)
        # If there is no PMW for the block then create it (creation happens on demand not eagerly)
        if pmw is None:
            pmw = self.addPMW(block)
        pmw.run_cache(query_id, block, budget)

    def get_execution_plan(self, query_id, blocks, budget):
        """
        For per-block-pmu all plans have this form: A(F(B1), F(B2), ... , F(Bn))
        """
        num_aggregations = len(blocks)
        plan = []
        splits = get_splits(blocks, num_aggregations)
        for split in splits:
            # print("split", split)
            for x in split:
                x = (x[0], x[-1])
                plan += [C(query_id, x, budget)]
                return A(plan)
        return None

