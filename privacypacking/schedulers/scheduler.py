from typing import List

from loguru import logger

from privacypacking.budget import Block, Task

# NOTE: ideally, we should be able to plug this class in Kubernetes,
# with just some thin wrappers to manage the CRD state and API calls.
# Would our current architecture work? Or is it not worth trying to be compatible?


class Scheduler:
    # TODO: What is config? A `Config` object?
    def __init__(self, tasks, blocks, config=None):
        self.tasks = tasks
        self.blocks = blocks
        self.config = config

        # TODO: manage the tasks state inside the scheduler. Pending, allocated, failed?
        self.allocated_tasks = {}

    def schedule(self) -> List[int]:
        """Takes some tasks from `self.tasks` and allocates them
        to some blocks from `self.blocks`.

        Modifies the budgets of the blocks inplace.

        Returns:
            List[int]: the ids of the tasks that were scheduled
        """
        pass

    def order(self) -> List[int]:
        pass

    def can_run(self, task):
        """
        A task can run only if we can allocate the demand budget
        for all the blocks requested
        """
        for block_id, demand_budget in task.budget_per_block.items():
            block = self.blocks[block_id]
            if not block.budget.can_allocate(demand_budget):
                return False
        return True

    def consume_budgets(self, task):
        """
        Updates the budgets of each block requested by the task
        """
        for block_id, demand_budget in task.budget_per_block.items():
            block = self.blocks[block_id]
            block.budget -= demand_budget

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def add_block(self, block: Block) -> None:
        if block.id in self.blocks:
            raise Exception("This block id is already present in the scheduler.")
        self.blocks.update({block.id: block})

    def update_allocated_tasks(self, allocated_task_ids: List[int]) -> None:
        """Pops allocated tasks from `self.tasks` and adds them to `self.allocated_tasks`

        Args:
            allocated_task_ids (List[int]): returned by `self.schedule()`
        """

        # TODO: this is quite horrible, we should change `self.task` (but retrocompatible)
        tasks = {task.id: task for task in self.tasks}
        for task_id in allocated_task_ids:
            self.allocated_tasks[task_id] = tasks.pop(task_id)

        self.tasks = list(tasks.values())