
from privacypacking.planner.planner import Planner
from privacypacking.cache.cache import A, R
from privacypacking.cache.utils import get_splits
import math

# This planner instance is naive - has only one option for a plan
class PerBlockPlanner(Planner):
    def __init__(self, cache, blocks):
        super().__init__(cache)
        self.blocks = blocks
        
    def get_execution_plan(self, query_id, blocks, budget):
        """
        For "per-block-planning" a plan has this form: A(R(B1), R(B2), ... , R(Bn))
        """
        num_aggregations = len(blocks) - 1
        plan = []
        split = get_splits(blocks, num_aggregations)[0]     # only one split in this case
        # print("split", split)
        for x in split:
            x = (x[0], x[-1])
            plan += [R(query_id, x, budget)]
        plan = A(plan)

        cost = self.cache.get_cost(plan, self.blocks)
        # print("Get cost of plan", plan, "cost", cost)

        if not math.isinf(cost):    # Get the cost of the plan - if infinite it's not eligible
            return plan
        return None




