from privacypacking.schedulers import simplex
from loguru import logger

from privacypacking.schedulers.budget_unlocking import (
    NBudgetUnlocking,
    TBudgetUnlocking,
)
from privacypacking.schedulers.scheduler import Scheduler
from privacypacking.schedulers.threshold_update_mechanisms import (
    ThresholdUpdateMechanism,
)
from privacypacking.schedulers.metrics import Metric
from privacypacking.schedulers.threshold_updating import ThresholdUpdating
from privacypacking.schedulers.utils import (
    BASIC_SCHEDULER,
    TASK_BASED_BUDGET_UNLOCKING,
    TIME_BASED_BUDGET_UNLOCKING,
    THRESHOLD_UPDATING,
    SIMPLEX,
)

# Loading them so that they get stored in globals()
from privacypacking.schedulers.metrics import (
    DominantShares,
    Fcfs,
    FlatRelevance,
    DynamicFlatRelevance,
    SquaredDynamicFlatRelevance,
    OverflowRelevance,
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
        return schedulers[config.scheduler_method]()
    else:
        metric = Metric.from_str(config.scheduler_metric)
        assert metric is not None

        # Some schedulers might need custom arguments
        if config.scheduler_method == BASIC_SCHEDULER:
            return schedulers[config.scheduler_method]()
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
