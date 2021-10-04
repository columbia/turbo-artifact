from typing import List

from privacypacking.schedulers.scheduler import Scheduler
from privacypacking.schedulers.threshold_update_mechanisms import (
    NaiveAverage,
    # QueueAverageDynamic,
)

"""
For all schedulers based on threshold updating approach 
"""


class ThresholdUpdating(Scheduler):
    def __init__(self, metric, scheduler_threshold_update_mechanism):
        super().__init__(metric)
        self.scheduler_threshold_update_mechanism = scheduler_threshold_update_mechanism

    def schedule_queue(
            self,
    ) -> List[int]:
        allocated_task_ids = []
        # Try and schedule tasks
        tasks = self.task_queue.tasks
        for task in tasks:
            task_cost = self.metric(task, self.blocks, tasks)
            # print(
            #     "Exceeds threshold?",
            #     self.task_queue.efficiency_threshold,
            #     task.get_efficiency(task_cost),
            #     "\n",
            #     task.budget,
            # )
            passed_threshold = False
            if self.task_queue.efficiency_threshold <= task.get_efficiency(task_cost):
                # print("Yes! :-)")
                passed_threshold = True
                if super().can_run(task):
                    self.allocate_task(task)
                    allocated_task_ids.append(task.id)

            # Re-calculate queue's threshold after observing the efficiency of the task
            self.post_update_queue_threshold(
                task.get_efficiency(task_cost), passed_threshold
            )
            # print("\nUpdate", self.task_queue.efficiency_threshold)
            # print("\n\n")

        return allocated_task_ids

    def post_update_queue_threshold(self, efficiency: float, can_run: bool) -> None:
        self.scheduler_threshold_update_mechanism.update_threshold(
            self.task_queue, efficiency, can_run
        )
