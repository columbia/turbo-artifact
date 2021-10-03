from typing import Type

from privacypacking.schedulers.scheduler import TaskQueue


class ThresholdUpdateMechanismException(Exception):
    pass


class ThresholdUpdateMechanism:
    @staticmethod
    def from_str(threshold_update_mechanism: str) -> Type["ThresholdUpdateMechanism"]:
        if threshold_update_mechanism in globals():
            return globals()[threshold_update_mechanism]
        else:
            raise ThresholdUpdateMechanismException(
                f"Unknown threshold_update_mechanism: {threshold_update_mechanism}"
            )

    @staticmethod
    def update_threshold(queue: TaskQueue, efficiency: float, passed_threshold: bool):
        pass


class NaiveAverage(ThresholdUpdateMechanism):
    @staticmethod
    def update_threshold(queue: TaskQueue, efficiency: float, passed_threshold: bool) -> None:
        queue.efficiency_threshold = (queue.efficiency_threshold + efficiency) / 2


class QueueAverageDynamic(ThresholdUpdateMechanism):
    @staticmethod
    def update_threshold(queue: TaskQueue, efficiency: float, can_run: bool) -> None:
        queue.efficiency_threshold = 0
        for task in queue.tasks:
            queue.efficiency_threshold += task.get_efficiency(task.cost)
        if can_run:
            queue.efficiency_threshold -= efficiency
        queue.efficiency_threshold /= len(queue.tasks) - 1


class QueueAverageStatic(ThresholdUpdateMechanism):
    @staticmethod
    def update_threshold(
            queue: TaskQueue, cost: float = None, passed_threshold: bool = None
    ) -> None:
        queue.efficiency_threshold = 0
        for task in queue.tasks:
            print(task.efficiency)
            queue.efficiency_threshold += task.get_efficiency(task.cost)
        queue.efficiency_threshold /= len(queue.tasks)
