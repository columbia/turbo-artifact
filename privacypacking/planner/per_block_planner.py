
from privacypacking.planner.planner import Planner
from privacypacking.cache.cache import A, R
from privacypacking.cache.utils import get_splits
from privacypacking.budget.block import HyperBlock
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

        if not math.isinf(self.get_cost(plan, self.blocks)):    # Get the cost of the plan - if infinite it's not eligible
            # Only one split for per-block - we stop here
            if len(plan) == 1:
                return plan[0]
            else:
                return A(plan)

        return None


# Cost model
def get_cost(plan, cache, blocks):     # Cost is either infinite or 0 in this implementation

    if isinstance(plan, A):     # Get cost of arguments/operators
        cost = sum([get_cost(x) for x in plan.l])

    elif isinstance(plan, R):   # Get cost of Run
        block_ids = list(range(plan.blocks[0], plan.blocks[-1] + 1))
        hyperblock = HyperBlock({key: blocks[key] for key in block_ids})
        
        if cache.get_entry(plan.query_id, hyperblock.id) is not None:
            return 0        # Already cached
        else:
            demand = {key: plan.budget for key in block_ids}
            if not hyperblock.can_run(demand):
                return math.inf     # This hyperblock does not have enough budget

            return 0     # Even if there is at least a little budget left in the hyperblock we assume the cost is 0 TODO: change this

    return cost



