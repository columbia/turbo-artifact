from threading import Lock
from typing import List

from privacypacking.budget import Block, Task
from privacypacking.simulator.tasks import TaskMessage

# NOTE: ideally, we should be able to plug this class in Kubernetes,
# with just some thin wrappers to manage the CRD state and API calls.
# Would our current architecture work? Or is it not worth trying to be compatible?


class Scheduler:
    def __init__(self):
        self.task_messages = []
        self.blocks = {}
        self.blocks_mutex = Lock()
        # TODO: manage the tasks state inside the scheduler. Pending, allocated, failed?
        self.allocated_tasks = {}

    def safe_get_num_blocks(self) -> int:
        self.blocks_mutex.acquire()
        num_blocks = len(self.blocks)
        self.blocks_mutex.release()
        return num_blocks

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
            if not block_id in self.blocks:
                return False
            block = self.blocks[block_id]
            if not block.budget.can_allocate(demand_budget):
                return False
        return True

    def consume_budgets(self, task) -> None:
        """
        Updates the budgets of each block requested by the task
        """
        for block_id, demand_budget in task.budget_per_block.items():
            block = self.blocks[block_id]
            block.budget -= demand_budget

    def add_task(self, task_message: TaskMessage) -> None:
        self.task_set_block_ids(task_message.task)
        self.task_messages.append(task_message)

        allocated_task_ids = self.schedule()
        self.update_allocated_tasks(allocated_task_ids)

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
        # TODO: this is quite horrible, we should change `self.task` (but retrocompatible)
        task_messages = {task_message.task.id: task_message for task_message in self.task_messages}
        for task_id in allocated_task_ids:
            self.task_messages[task_id].allocated_resources_event.succeed()
            self.allocated_tasks[task_id] = self.task_messages.pop(task_id).task
        self.task_messages = list(task_messages.values())

    def task_set_block_ids(self, task: Task) -> None:
        # Ask the stateful scheduler to set the block ids of the task according to the task's constraints

        # Acquire the lock before trying to get an instance of the current blocks state
        # otherwise, other blocks might be added while selecting blocks for a task
        self.blocks_mutex.acquire()
        try:
            selected_block_ids = task.block_selection_policy.select_blocks(
                blocks=self.blocks, task_blocks_num=task.n_blocks
            )
        finally:
            self.blocks_mutex.release()

        assert selected_block_ids is not None
        task.set_budget_per_block(selected_block_ids)
