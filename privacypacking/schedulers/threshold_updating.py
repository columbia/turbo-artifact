from typing import List
from privacypacking.budget import Task
from privacypacking.schedulers import Scheduler
from privacypacking.schedulers.scheduler import TaskQueue
from privacypacking.schedulers.threshold_update_mechanisms import (
    QueueAverageDynamic,
    QueueAverageStatic,
    NaiveAverage,
)

"""
For all schedulers based on threshold updating approach 

"""


class ThresholdUpdating(Scheduler):
    def __init__(
            self, env, number_of_queues, metric, scheduler_threshold_update_mechanism
    ):
        super().__init__(env, number_of_queues, metric)
        self.scheduler_threshold_update_mechanism = scheduler_threshold_update_mechanism

    def schedule_queue(self, queue: TaskQueue) -> List[int]:
        """Takes some tasks from `self.tasks` and allocates them
        to some blocks from `self.blocks`.
        Modifies the budgets of the blocks inplace.
        Returns:
            List[int]: the ids of the tasks that were scheduled
        """
        tasks = queue.tasks
        allocated_task_ids = []
        # Task sorted by 'metric'
        sorted_tasks = self.order(tasks)

        # Do some static pre-calculation of queue's threshold before trying to schedule
        self.pre_update_queue_threshold(queue)

        # Try and schedule tasks
        for task in sorted_tasks:
            if self.can_run(task):
                self.allocate_task(task)
                allocated_task_ids.append(task.id)
        return allocated_task_ids

    def can_run(self, task: Task) -> bool:
        can_run = False
        queue = self.get_queue_from_task(task)

        print("Can run?", queue.cost_threshold, task.cost, "\n", task.budget)
        if queue.cost_threshold >= task.cost:
            can_run = super().can_run(task)
        if not can_run:
            print("No!")

        # Re-calculate queue's threshold after observing the cost of the task
        self.post_update_queue_threshold(queue, task.cost, can_run)

        print("\nUpdate", queue.cost_threshold)
        print("\n\n")
        return can_run

    def pre_update_queue_threshold(self, queue: TaskQueue) -> None:
        if self.scheduler_threshold_update_mechanism == QueueAverageStatic:
            self.scheduler_threshold_update_mechanism.update_threshold(queue)

    def post_update_queue_threshold(
            self, queue: TaskQueue, cost: float, can_run: bool
    ) -> None:
        if self.scheduler_threshold_update_mechanism != QueueAverageStatic:
            self.scheduler_threshold_update_mechanism.update_threshold(queue, cost, can_run)
