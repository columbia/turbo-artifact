import time
from typing import List, Tuple

from loguru import logger
from simpy import Event

from privacypacking.budget import Block, Task
from privacypacking.budget.block_selection import NotEnoughBlocks
from privacypacking.schedulers.utils import ALLOCATED, FAILED, PENDING


class TaskQueue:
    def __init__(self):
        self.tasks = []
        self.efficiency_threshold = 100


class TasksInfo:
    def __init__(self):
        self.allocated_tasks = {}
        self.allocated_resources_events = {}
        self.tasks_status = {}
        self.tasks_scheduling_time = {}

    def dump(self):
        tasks_info = {"allocated_tasks": {}, "tasks_scheduling_time": {}}
        for task_id, task in self.allocated_tasks.items():
            tasks_info["allocated_tasks"][task_id] = task
            tasks_info["tasks_scheduling_time"][task_id] = self.tasks_scheduling_time[
                task_id
            ]
        return tasks_info


class Scheduler:
    def __init__(self, metric=None):
        self.metric = metric
        self.task_queue = TaskQueue()
        self.blocks = {}
        self.tasks_info = TasksInfo()

    def consume_budgets(self, task):
        """
        Updates the budgets of each block requested by the task
        """
        for block_id, demand_budget in task.budget_per_block.items():
            block = self.blocks[block_id]
            block.budget -= demand_budget

    def allocate_task(self, task: Task) -> None:
        """
        Updates the budgets of each block requested by the task and cleans up scheduler's state
        """
        # Consume_budgets
        self.consume_budgets(task)
        # Clean/update scheduler's state
        self.tasks_info.tasks_status[task.id] = ALLOCATED
        self.tasks_info.allocated_resources_events[task.id].succeed()
        del self.tasks_info.allocated_resources_events[task.id]
        self.tasks_info.tasks_scheduling_time[task.id] = (
            time.time() - self.tasks_info.tasks_scheduling_time[task.id]
        )
        self.tasks_info.allocated_tasks[task.id] = task
        self.task_queue.tasks.remove(task)  # Todo: this takes linear time -> optimize

    def schedule_queue(self) -> List[int]:
        """Takes some tasks from `self.tasks` and allocates them
        to some blocks from `self.blocks`.
        Modifies the budgets of the blocks inplace.
        Returns:
            List[int]: the ids of the tasks that were scheduled
        """
        allocated_task_ids = []
        # Run until scheduling cycle ends
        converged = False
        while not converged:
            sorted_tasks = self.order(self.task_queue.tasks)
            converged = True
            for task in sorted_tasks:
                if self.can_run(task):
                    self.allocate_task(task)
                    allocated_task_ids.append(task.id)
                    if self.metric().is_dynamic():
                        converged = False
                        break
        return allocated_task_ids

    def add_task(self, task_message: Tuple[Task, Event]):
        (task, allocated_resources_event) = task_message
        try:
            self.task_set_block_ids(task)
        except NotEnoughBlocks as e:
            logger.warning(
                f"{e}\n Skipping this task as it can't be allocated. Will not count in the total number of tasks?"
            )
            self.tasks_info.tasks_status[task.id] = FAILED
            return

        # Update tasks_info
        self.tasks_info.tasks_status[task.id] = PENDING
        self.tasks_info.allocated_resources_events[task.id] = allocated_resources_event
        self.tasks_info.tasks_scheduling_time[task.id] = time.time()
        self.task_queue.tasks.append(task)

    def add_block(self, block: Block) -> None:
        if block.id in self.blocks:
            raise Exception("This block id is already present in the scheduler.")
        self.blocks.update({block.id: block})

    def get_num_blocks(self) -> int:
        num_blocks = len(self.blocks)
        return num_blocks

    def order(self, tasks: List[Task]) -> List[Task]:
        """Sorts the tasks by metric"""

        def task_key(task):
            return self.metric.apply(task, self.blocks, tasks)

        return sorted(tasks, reverse=True, key=task_key)

    def can_run(self, task: Task) -> bool:
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

    def task_set_block_ids(self, task: Task) -> None:
        # Ask the stateful scheduler to set the block ids of the task according to the task's constraints
        # try:
        selected_block_ids = task.block_selection_policy.select_blocks(
            blocks=self.blocks, task_blocks_num=task.n_blocks
        )
        # except NotEnoughBlocks as e:
        #     logger.warning(e)
        #     logger.warning(
        #         "Setting block ids to -1, the task will never be allocated.\n Should we count it in the total number of tasks?"
        #     )
        #     selected_block_ids = [-1]
        assert selected_block_ids is not None
        task.set_budget_per_block(selected_block_ids)
