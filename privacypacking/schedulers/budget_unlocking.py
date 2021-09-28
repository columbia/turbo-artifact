from typing import Tuple, Any

from simpy import Event

from privacypacking.budget import Block, Task
from privacypacking.budget import ZeroCurve
from privacypacking.schedulers import Scheduler

"""
For all schedulers based on gradually unlocking budget

"""


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
