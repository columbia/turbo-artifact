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
    def update_threshold(
            queue: TaskQueue, efficiency: float, passed_threshold: bool
    ) -> None:
        queue.efficiency_threshold = (queue.efficiency_threshold + efficiency) / 2

# class QueueAverageDynamic(ThresholdUpdateMechanism):
#     past_efficiencies = []
#
#     @staticmethod
#     def update_threshold(queue: TaskQueue, efficiency: float, passed_threshold: bool) -> None:
#         QueueAverageDynamic.past_efficiencies.append(efficiency)
#         queue.efficiency_threshold = 0
#         r = min(len(QueueAverageDynamic.past_efficiencies), 1)
#         for i in range(r):        # See ten last efficiencies
#             queue.efficiency_threshold += QueueAverageDynamic.past_efficiencies[r-i-1]
#         # if r>0:
#         queue.efficiency_threshold /= r
