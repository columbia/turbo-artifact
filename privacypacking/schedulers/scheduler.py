from simpy import Event
from threading import Lock
from typing import List, Tuple, Type, Any
from privacypacking.budget import Block, Task
from privacypacking.schedulers.utils import PENDING, ALLOCATED


# NOTE: ideally, we should be able to plug this class in Kubernetes,
# with just some thin wrappers to manage the CRD state and API calls.
# Would our current architecture work? Or is it not worth trying to be compatible?
# Kelly: I think we might need to fully remove simpy for the realistic setting of kubernetes
# and convert this into a real distributed system rather than a discrete simulation


class TaskQueue:
    def __init__(self, t):
        self.tasks = []
        self.time_window = t
        self.cost_threshold = 0

class TasksInfo:
    def __init__(self):
        self.allocated_tasks = {}
        self.allocated_resources_events = {}
        self.tasks_status = {}
        self.tasks_priority = {}


class Scheduler:
    def __init__(self, env, metric=None):
        self.env = env
        self.metric = metric
        self.task_queues = {}
        self.blocks = {}
        self.blocks_mutex = Lock()
        self.tasks_info = TasksInfo()

    def get_queue_from_task(self, task):
        priority_num = self.tasks_info.tasks_priority[task.id]
        return self.task_queues[priority_num]

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
        self.tasks_info.allocated_tasks[task.id] = task
        self.get_queue_from_task(task).tasks.remove(task)  # Todo: this takes linear time -> optimize

    def schedule(self, tasks: List[Task]) -> List[int]:
        """Takes some tasks from `self.tasks` and allocates them
        to some blocks from `self.blocks`.
        Modifies the budgets of the blocks inplace.
        Returns:
            List[int]: the ids of the tasks that were scheduled
        """
        allocated_task_ids = []
        # Task sorted by 'metric'
        sorted_tasks = self.order(tasks)
        # Try and schedule tasks
        for task in sorted_tasks:
            if self.can_run(task):
                self.allocate_task(task)
                allocated_task_ids.append(task.id)
        return allocated_task_ids

    def schedule_queue(self, queue: TaskQueue) -> List[int]:
        return self.schedule(queue.tasks)

    def wait_and_schedule_queue(self, queue: TaskQueue) -> List[int]:
        while True:
            # Waits for "time_window" units of time
            yield self.env.timeout(queue.time_window)
            # Try and schedule the tasks existing in the queue
            return self.schedule(queue.tasks)

    def add_task(self, task_message: Tuple[Task, Event]) -> Tuple[Any, bool]:
        (task, allocated_resources_event) = task_message
        self.task_set_block_ids(task)

        priority_num = 0  # fixed for now (should depend on task's profit)
        t = 0  # fixed for now (should depend on the priority); we allow only one queue

        # Update tasks_info
        self.tasks_info.tasks_priority[task.id] = priority_num
        self.tasks_info.tasks_status[task.id] = PENDING
        self.tasks_info.allocated_resources_events[task.id] = allocated_resources_event

        is_new_queue = False
        # If there is no queue for that priority yet create one
        if priority_num not in self.task_queues:
            is_new_queue = True
            self.task_queues[priority_num] = TaskQueue(t)

        self.task_queues[priority_num].tasks.append(task)
        return self.task_queues[priority_num], is_new_queue

    def add_block(self, block: Block) -> None:
        if block.id in self.blocks:
            raise Exception("This block id is already present in the scheduler.")
        self.blocks.update({block.id: block})

    def safe_add_block(self, block: Block) -> None:
        self.blocks_mutex.acquire()
        try:
            self.add_block(block)
        finally:
            self.blocks_mutex.release()

    def safe_get_num_blocks(self) -> int:
        self.blocks_mutex.acquire()
        num_blocks = len(self.blocks)
        self.blocks_mutex.release()
        return num_blocks

    def order(self, tasks: List[Task]) -> List[Task]:
        """Sorts the tasks by metric"""
        return self.metric(tasks, self.blocks)

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
