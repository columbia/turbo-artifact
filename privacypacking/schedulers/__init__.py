from privacypacking.schedulers.metrics import (
    dominant_shares,
    fcfs,
    overflow_relevance,
    flat_relevance,
    round_robins,
)
from privacypacking.schedulers import greedy_heuristics, simplex
from privacypacking.schedulers.scheduler import Scheduler
from privacypacking.schedulers.budget_unlocking import BudgetUnlocking
from privacypacking.schedulers.threshold_updating import ThresholdUpdating


def get_scheduler(env, config) -> Scheduler:
    schedulers = {
        "basic_scheduler": Scheduler,
        "budget_unlocking": BudgetUnlocking,
        "threshold_updating": ThresholdUpdating,
        "simplex": simplex.Simplex,
        # todo: the following should be metrics; not schedulers
        "OfflineDPF": greedy_heuristics.OfflineDPF,
        "FlatRelevance": greedy_heuristics.FlatRelevance,
        "OverflowRelevance": greedy_heuristics.OverflowRelevance,
    }
    metric = None
    if config.scheduler_metric in globals():
        metric = globals()[config.scheduler_metric]
    assert metric is not None
    # Some schedulers might need custom arguments
    if config.scheduler_method == "basic_scheduler":
        return schedulers[config.scheduler_method](env, config.number_of_queues, metric)
    elif config.scheduler_method == "budget_unlocking":
        return schedulers[config.scheduler_method](
            env, config.number_of_queues, metric, config.scheduler_N
        )
    elif config.scheduler_method == "threshold_updating":
        return schedulers[config.scheduler_method](env, config.number_of_queues, metric)
    elif config.scheduler_method == "simplex":
        return schedulers[config.scheduler_method](env)
    else:
        return schedulers[config.scheduler_method](env, metric)
