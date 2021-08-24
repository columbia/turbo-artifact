from typing import List

from loguru import logger

from privacypacking.budget import Block, Task
from threading import Lock

# NOTE: ideally, we should be able to plug this class in Kubernetes,
# with just some thin wrappers to manage the CRD state and API calls.
# Would our current architecture work? Or is it not worth trying to be compatible?


class Scheduler:
    def __init__(self, tasks, blocks):
        self.tasks = {task.id: task for task in tasks}
        self.blocks = {block.id: block for block in blocks}
        self.blocks_mutex = Lock()

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

    def safe_schedule(self) -> List[int]:
        self.blocks_mutex.acquire()
        try:
            allocated_ids = self.schedule()
        finally:
            self.blocks_mutex.release()
        return allocated_ids

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
        self.tasks.update({task.id: task})

    def add_block(self, block: Block) -> None:
        if block.id in self.blocks:
            raise Exception("This block id is already present in the scheduler.")
        self.blocks.update({block.id: block})

    def safe_add_block(self, block: Block) -> None:
        self.blocks_mutex.acquire()
        try:
            if block.id in self.blocks:
                raise Exception("This block id is already present in the scheduler.")
            self.blocks.update({block.id: block})
        finally:
            self.blocks_mutex.release()

    def update_allocated_tasks(self, allocated_task_ids: List[int]) -> None:
        """Pops allocated tasks from `self.tasks` and adds them to `self.allocated_tasks`

        Args:
            allocated_task_ids (List[int]): returned by `self.schedule()`
        """
        for task_id in allocated_task_ids:
            self.allocated_tasks[task_id] = self.tasks.pop(task_id)

    def safe_select_block_ids(self, num_blocks, policy_func):
        self.blocks_mutex.acquire()
        try:
            selected_block_ids = policy_func(
                blocks=self.blocks, task_blocks_num=num_blocks
            ).select_blocks()
        finally:
            self.blocks_mutex.release()

        assert selected_block_ids is not None
        return selected_block_ids
