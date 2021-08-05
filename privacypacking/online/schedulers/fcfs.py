from privacypacking.online.schedulers.scheduler import Scheduler
from privacypacking.utils.utils import *

# TODO: see if we can reuse the greedy heuristic here
# (FCFS is a greedy heuristic with no heuristic)


class FCFS(Scheduler):
    """
    Schedule by prioritizing the tasks that come first
    """

    def __init__(self, tasks, blocks, config=None):
        super().__init__(tasks, blocks, config)

    def schedule(self):
        allocation = [False] * len(self.tasks)

        # Read them by order
        for i, task in enumerate(self.tasks):
            for block_id, demand_budget in task.budget_per_block:
                block = get_block_by_block_id(self.blocks, block_id)
                # Remaining block budget
                block_budget = block.budget
                # Demand for this block
                demand_budget = task.budget_per_block[block_id]

                # todo: lock block
                allocation[i] = block_budget.allocate_budget(demand_budget)

        return allocation
