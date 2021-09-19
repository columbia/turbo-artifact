from privacypacking.schedulers import dpf, fcfs, greedy_heuristics, simplex
from privacypacking.schedulers.scheduler import Scheduler


def get_scheduler(env, config) -> Scheduler:
    schedulers = {
        "fcfs": fcfs.FCFS,
        "dpf": dpf.DPF,
        "simplex": simplex.Simplex,
        "OfflineDPF": greedy_heuristics.OfflineDPF,
        "FlatRelevance": greedy_heuristics.FlatRelevance,
        "OverflowRelevance": greedy_heuristics.OverflowRelevance,
    }
    # Some schedulers might need some additional arguments
    if config.scheduler_name == "dpf":
        return schedulers[config.scheduler_name](env, config.scheduler_N)
    else:
        return schedulers[config.scheduler_name](env)
