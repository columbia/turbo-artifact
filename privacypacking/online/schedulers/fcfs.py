from privacypacking.online.schedulers.scheduler import Scheduler
from privacypacking.utils.utils import *

# TODO: see if we can reuse the greedy heuristic here
# (FCFS is a greedy heuristic with no heuristic)


class FCFS(Scheduler):
    """
    Schedule by prioritizing the tasks that come first
    """

    def __init__(self, tasks, blocks):
        super().__init__(tasks, blocks)

    def schedule(self):
        allocation = [False] * len(self.tasks)

        # Read them by order
        for i, task in enumerate(self.tasks):
            for block_id in task.block_ids:
                block = get_block_by_block_id(self.blocks, block_id)
                # Remaining block budget
                block_budget = block.budget
                # Demand for this block
                demand_budget = task.budget_per_block[block_id]

                # todo: lock block
                # There must exist at least one order in the block's budget
                # that is smaller or equal to the corresponding order of the demand budget
                diff = block_budget - demand_budget
                max_order = max(diff.orders.values())
                if max_order >= 0:
                    block.budget = diff
                    allocation[i] = True

        return allocation
