from typing import Any, Tuple

from simpy import Event

from privacypacking.budget import Block, Budget, Task, ZeroCurve
from privacypacking.schedulers.scheduler import Scheduler

"""
For all schedulers based on gradually unlocking budget

"""


class UnlockingBlock(Block):
    def __init__(self, id, budget, n=1):
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


class NBudgetUnlocking(Scheduler):
    """N-unlocking: unlocks some budget every time a new task arrives."""

    def __init__(self, env, number_of_queues, metric, n):
        super().__init__(env, number_of_queues, metric)
        self.n = n
        assert self.n is not None

    def add_task(self, task_message: Tuple[Task, Event]) -> Tuple[Any, bool]:
        queue, is_new_queue = super().add_task(task_message)
        self.unlock_block_budgets(queue.tasks)
        return queue, is_new_queue

    def safe_add_block(self, block: Block) -> None:
        unlocking_block = UnlockingBlock(block.id, block.budget, self.n)
        super().add_block(unlocking_block)

    def unlock_block_budgets(self, tasks):
        new_task = tasks[-1]
        # Unlock budget only for blocks demanded by the last task
        for block_id in new_task.budget_per_block.keys():
            # Unlock budget for each alpha
            self.blocks[block_id].unlock_budget()

    def can_run(self, task):
        """
        A task can run only if we can allocate the demand budget
        for all the blocks requested, by using only unlocked budget
        (i.e. after allocating the task, for each block there is one alpha
        that is below the unlocked budget)
        """
        for block_id, demand_budget in task.budget_per_block.items():
            block = self.blocks[block_id]
            allocated_budget = block.initial_budget - block.budget
            available_budget = block.unlocked_budget - allocated_budget
            if not available_budget.can_allocate(demand_budget):
                return False
        return True


# TODO: Write a T-Unlocking scheduler too


# TODO: sublass of Block?
class WrapperBlock:
    """
    A wrapper for the traditional-block.
    """

    def __init__(self, block, n):
        self.id = block.id
        self.budget = (
            ZeroCurve()
        )  # Will be gradually unlocking budget till we reach full capacity
        self.allocated_budget = ZeroCurve()  # Budget currently allocated by tasks
        self.block = block
        self.fair_share = self.block.initial_budget / n

    def unlock_budget(self):
        self.budget = self.budget.add_with_threshold(
            self.fair_share, self.block.initial_budget
        )
        # print("\n\nFair Share \n", self.fair_share)
        # print("\nUpdate budget\n", self.budget)
        # print("\nTotal budget capacity\n", self.block.initial_budget)
        # print("\n\n")


class BudgetUnlocking(Scheduler):
    # Static variable
    # TODO: why??
    wrapper_blocks = {}

    def __init__(self, env, number_of_queues, metric, n):
        super().__init__(env, number_of_queues, metric)
        self.n = n
        assert self.n is not None

    def add_task(self, task_message: Tuple[Task, Event]) -> Tuple[Any, bool]:
        queue, is_new_queue = super().add_task(task_message)
        self.unlock_block_budgets(queue.tasks)
        return queue, is_new_queue

    def safe_add_block(self, block: Block) -> None:
        self.blocks_mutex.acquire()
        try:
            if block.id in self.blocks:
                raise Exception("This block id is already present in the scheduler.")
            self.blocks.update({block.id: block})
            BudgetUnlocking.wrapper_blocks[block.id] = WrapperBlock(block, self.n)
        finally:
            self.blocks_mutex.release()

    def unlock_block_budgets(self, tasks):
        new_task = tasks[-1]
        for block_id in new_task.budget_per_block.keys():
            wrapper_block = BudgetUnlocking.wrapper_blocks[block_id]
            # Unlock budget for each alpha
            wrapper_block.unlock_budget()

    def can_run(self, task):
        """
        A task can run only if we can allocate the demand budget
        for all the blocks requested
        """
        for block_id, demand_budget in task.budget_per_block.items():
            wrapper_block = BudgetUnlocking.wrapper_blocks[block_id]
            available_budget = wrapper_block.budget - wrapper_block.allocated_budget
            if not available_budget.can_allocate(demand_budget):
                return False
        return True

    def consume_budgets(self, task):
        """
        Updates the budgets of each block requested by the task
        """
        for block_id, demand_budget in task.budget_per_block.items():
            wrapper_block = BudgetUnlocking.wrapper_blocks[block_id]
            wrapper_block.allocated_budget += demand_budget
            # Consume traditional block's budget as well
            wrapper_block.block.budget -= demand_budget
