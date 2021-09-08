from typing import Type

from privacypacking.schedulers import dpf, fcfs, greedy_heuristics, simplex
from privacypacking.schedulers.scheduler import Scheduler

# from privacypacking.utils.utils import DPF, FCFS, SIMPLEX


def get_scheduler_class(scheduler_name: str) -> Type[Scheduler]:
    schedulers = {
        "fcfs": fcfs.FCFS,
        "dpf": dpf.DPF,
        "simplex": simplex.Simplex,
        "OfflineDPF": greedy_heuristics.OfflineDPF,
        "FlatRelevance": greedy_heuristics.FlatRelevance,
        "OverflowRelevance": greedy_heuristics.OverflowRelevance,
    }
    return schedulers[scheduler_name]
