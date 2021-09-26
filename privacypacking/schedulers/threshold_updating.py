from typing import List
from privacypacking.budget import Task
from privacypacking.schedulers import Scheduler
from privacypacking.schedulers.scheduler import TaskQueue
from privacypacking.schedulers.utils import (
    QUEUE_AVERAGE_STATIC,
    QUEUE_AVERAGE_DYNAMIC,
    NAIVE_AVERAGE,
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
        if self.scheduler_threshold_update_mechanism == globals()[QUEUE_AVERAGE_STATIC]:
            self.scheduler_threshold_update_mechanism(queue)
        return self.schedule(queue.tasks)

    def can_run(self, task) -> bool:
        can_run = False
        queue = self.get_queue_from_task(task)
        print("Can run?", queue.cost_threshold, task.cost, "\n", task.budget)
        if queue.cost_threshold >= task.cost:
            can_run = super().can_run(task)
        if not can_run:
            print("No!")

        self.update_queue_threshold(queue, task.cost, can_run)
        print("\nUpdate", queue.cost_threshold)
        print("\n\n")
        return can_run

    def update_queue_threshold(
            self, queue: TaskQueue, cost: float, can_run: bool
    ) -> None:
        if self.scheduler_threshold_update_mechanism != globals()[QUEUE_AVERAGE_STATIC]:
            self.scheduler_threshold_update_mechanism(queue, cost, can_run)
