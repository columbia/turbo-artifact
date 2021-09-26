from privacypacking.budget import Task
from privacypacking.schedulers import Scheduler

"""
For all schedulers based on threshold updating approach 

"""


class ThresholdUpdating(Scheduler):
    def __init__(self, env, metric):
        super().__init__(env, metric)

    def can_run(self, task):
        queue = self.get_queue_from_task(task)
        if queue.cost_threshold <= self.metric(task, self.blocks):
            return super().can_run(task)
        return False

    def update_threshold(self):
        pass

    def allocate_task(self, task: Task) -> None:
        self.update_threshold()
        super().allocate_task(task)
