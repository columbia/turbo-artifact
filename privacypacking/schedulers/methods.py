from loguru import logger

from privacypacking.schedulers import simplex
from privacypacking.schedulers.budget_unlocking import (
    NBudgetUnlocking,
    TBudgetUnlocking,
)

# Loading them so that they get stored in globals()
from privacypacking.schedulers.metrics import (
    DominantShares,
    DynamicFlatRelevance,
    Fcfs,
    FlatRelevance,
    Metric,
    OverflowRelevance,
    SquaredDynamicFlatRelevance,
)
from privacypacking.schedulers.scheduler import Scheduler
from privacypacking.schedulers.threshold_update_mechanisms import (
    ThresholdUpdateMechanism,
)
from privacypacking.schedulers.threshold_updating import ThresholdUpdating
from privacypacking.schedulers.utils import (
    BASIC_SCHEDULER,
    SIMPLEX,
    TASK_BASED_BUDGET_UNLOCKING,
    THRESHOLD_UPDATING,
    TIME_BASED_BUDGET_UNLOCKING,
)


def get_scheduler(config, env) -> Scheduler:
    schedulers = {
        BASIC_SCHEDULER: Scheduler,
        TASK_BASED_BUDGET_UNLOCKING: NBudgetUnlocking,
        TIME_BASED_BUDGET_UNLOCKING: TBudgetUnlocking,
        THRESHOLD_UPDATING: ThresholdUpdating,
        SIMPLEX: simplex.Simplex,
    }
    if config.scheduler_method == SIMPLEX:
        if config.scheduler_solver:
            return schedulers[config.scheduler_method](solver=config.scheduler_solver)
        else:
            schedulers[config.scheduler_method]()
    else:
        metric = Metric.from_str(config.scheduler_metric)
        assert metric is not None

        # Some schedulers might need custom arguments
        if config.scheduler_method == BASIC_SCHEDULER:
            return schedulers[config.scheduler_method](metric)
        elif config.scheduler_method == TASK_BASED_BUDGET_UNLOCKING:
            return schedulers[config.scheduler_method](metric, config.scheduler_N)
        elif config.scheduler_method == TIME_BASED_BUDGET_UNLOCKING:
            return schedulers[config.scheduler_method](
                metric,
                config.scheduler_N,
                config.scheduler_budget_unlocking_time,
                env,
            )
        elif config.scheduler_method == THRESHOLD_UPDATING:
            scheduler_threshold_update_mechanism = ThresholdUpdateMechanism.from_str(
                config.scheduler_threshold_update_mechanism
            )
            return schedulers[config.scheduler_method](
                metric,
                scheduler_threshold_update_mechanism,
            )
        else:
            logger.error(f"No such scheduler exists: {config.scheduler_method}")
            exit()
