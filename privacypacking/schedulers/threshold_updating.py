from privacypacking.budget import Task
from privacypacking.schedulers import Scheduler
from privacypacking.schedulers.scheduler import TaskQueue

"""
For all schedulers based on threshold updating approach 

"""


class ThresholdUpdating(Scheduler):
    def __init__(self, env, number_of_queues, metric):
        super().__init__(env, number_of_queues, metric)

    def can_run(self, task) -> bool:
        can_run = False
        queue = self.get_queue_from_task(task)
        if queue.cost_threshold <= task.cost:
            # print("cant run", queue.cost_threshold, task.cost)
            can_run = super().can_run(task)

        self.update_queue_threshold(queue, task.cost, can_run)
        # print("update", queue.cost_threshold)
        return can_run

    def update_queue_threshold(
            self, queue: TaskQueue, cost: float, can_run: bool
    ) -> None:
        self.naive_average(queue, cost)

    def naive_average(self, queue: TaskQueue, cost: float) -> None:
        queue.cost_threshold = (queue.cost_threshold + cost) / 2

    def allocate_task(self, task: Task) -> None:
        super().allocate_task(task)
