from typing import Tuple

from simpy import Event

from privacypacking.budget import Block, Budget, Task, ZeroCurve
from privacypacking.schedulers.scheduler import Scheduler

"""
For all schedulers based on gradually unlocking budget
"""


class UnlockingBlock(Block):
    def __init__(self, id: int, budget: Budget, n: int = 1):
        super().__init__(id, budget)
        self.unlocked_budget = (
            ZeroCurve()
        )  # Will be gradually unlocking budget till we reach full capacity
        self.fair_share = self.initial_budget / n

    def unlock_budget(self, budget: Budget = None):
        """Updates `self.unlocked_budget`. Fair share by default, but can use dynamic values too."""
        self.unlocked_budget = self.unlocked_budget.add_with_threshold(
            budget if budget else self.fair_share, self.initial_budget
        )
        # print("\n\nFair Share \n", self.fair_share)
        # print("\nUpdate budget\n", self.budget)
        # print("\nTotal budget capacity\n", self.block.initial_budget)
        # print("\n\n")


class NBudgetUnlocking(Scheduler):
    """N-unlocking: unlocks some budget every time a new task arrives."""

    def __init__(self, metric, n):
        super().__init__(metric)
        self.n = n
        assert self.n is not None

    def add_task(self, task_message: Tuple[Task, Event]):
        super().add_task(task_message)
        self.unlock_block_budgets(self.task_queue.tasks)

    def add_block(self, block: Block) -> None:
        unlocking_block = UnlockingBlock(block.id, block.budget, self.n)
        super().add_block(unlocking_block)

    def unlock_block_budgets(self, tasks):
        new_task = tasks[-1]
        # Unlock budget only for blocks demanded by the last task
        for block_id in new_task.budget_per_block.keys():
            # Unlock budget for each alpha
            self.blocks[block_id].unlock_budget()

    def can_run(self, task):
        for block_id, demand_budget in task.budget_per_block.items():
            block = self.blocks[block_id]
            allocated_budget = block.initial_budget - block.budget
            available_budget = block.unlocked_budget - allocated_budget
            if not available_budget.can_allocate(demand_budget):
                return False
        return True


# TODO: Write a T-Unlocking scheduler too
