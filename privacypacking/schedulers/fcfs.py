from privacypacking.schedulers.scheduler import Scheduler
from privacypacking.budget.task import Task
from typing import List


# TODO: see if we can reuse the greedy heuristic here
# (FCFS is a greedy heuristic with no heuristic)


class FCFS(Scheduler):
    """
    Schedule by prioritizing the tasks that come first
    """

    def __init__(self, env):
        super().__init__(env)

    def schedule(self, tasks: List[Task]):
        allocated_task_ids = []

        # Read them by order
        for i, task in enumerate(tasks):
            # self.task_set_block_ids(task)
            if self.can_run(task):
                self.allocate_task(task)
                allocated_task_ids.append(task.id)

        return allocated_task_ids
