from privacypacking.planner.planner import Planner
from privacypacking.cache.cache import A, R
import math

class NoPlanner(Planner):
    def __init__(self, cache, blocks):
        super().__init__(cache)
        self.blocks = blocks

    def get_execution_plan(self, query_id, blocks, budget):
        bs_tuple = (blocks[0], blocks[-1])
        plan = A([R(query_id=query_id, blocks=bs_tuple, budget=budget)])
        cost = self.cache.get_cost(plan, self.blocks)
        if not math.isinf(cost):    # Get the cost of the plan - if infinite it's not eligible
            return plan
        return None