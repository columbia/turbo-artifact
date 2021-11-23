import time
from typing import List, Tuple

from loguru import logger
from simpy import Event

from privacypacking.budget import Task
from privacypacking.schedulers.scheduler import Scheduler
from privacypacking.schedulers.threshold_update_mechanisms import (  # QueueAverageDynamic,
    NaiveAverage,
)

"""
For all schedulers based on threshold updating approach 
"""


class ThresholdUpdating(Scheduler):
    def __init__(self, metric, scheduler_threshold_update_mechanism):
        super().__init__(metric)
        self.scheduler_threshold_update_mechanism = scheduler_threshold_update_mechanism

    def schedule_queue(self) -> List[int]:
        print("\n\nSCHEDULING...")

        allocated_task_ids = []
        # Run until scheduling cycle ends
        converged = False
        while not converged:
            sorted_tasks = self.order(self.task_queue.tasks)

            # print("sorting...")
            converged = True
            for task in sorted_tasks:
                # Update efficiency threshold after observing the efficiency of the task
                self.update_efficiency_threshold(task.get_efficiency(task.cost))
                print(
                    f"\nTask {task.id}, Update: {self.task_queue.efficiency_threshold}"
                )
                print(
                    f"Exceeds {task.id} threshold?",
                    self.task_queue.efficiency_threshold,
                    task.get_efficiency(task.cost),
                    "\n",
                    task.budget,
                )

                if (
                    task.get_efficiency(task.cost)
                    >= self.task_queue.efficiency_threshold
                ):
                    print("Yes! :-)\n")
                    if self.can_run(task):
                        self.allocate_task(task)
                        allocated_task_ids.append(task.id)
                        if self.metric().is_dynamic():
                            converged = False
                            break
        return allocated_task_ids

    def update_efficiency_threshold(self, task_efficiency: float) -> None:
        self.scheduler_threshold_update_mechanism.update_threshold(
            self.task_queue, task_efficiency
        )
