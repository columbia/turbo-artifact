from loguru import logger

from privacypacking.schedulers import simplex
from privacypacking.schedulers.budget_unlocking import (
    NBudgetUnlocking,
    TBudgetUnlocking,
)

# Loading them so that they get stored in globals()
from privacypacking.schedulers.metrics import Metric
from privacypacking.schedulers.scheduler import Scheduler


def initialize_scheduler(config, env) -> Scheduler:

    scheduler_spec = config.omegaconf.scheduler
    logger.info(
        f"Initializing scheduler with spec: {scheduler_spec} {config.omegaconf}"
    )

    # TODO: integrate Simplex as a regular metric, nothing special about it.
    if scheduler_spec.method == "offline" and scheduler_spec.metric == "simplex":
        return simplex.Simplex(simulator_config=config.omegaconf)

    metric = Metric.from_str(
        scheduler_spec.metric, metric_config=config.omegaconf.metric
    )

    if scheduler_spec.method == "offline":
        return Scheduler(metric, simulator_config=config.omegaconf)

    if config.omegaconf.scheduler.method == "batch":
        return TBudgetUnlocking(
            metric,
            env,
            simulator_config=config.omegaconf,
        )

    raise ValueError("Unknown scheduler.")