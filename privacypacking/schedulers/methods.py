from privacypacking.schedulers import simplex
from privacypacking.schedulers.budget_unlocking import NBudgetUnlocking
from privacypacking.schedulers.scheduler import Scheduler
from privacypacking.schedulers.threshold_update_mechanisms import (
    ThresholdUpdateMechanism,
)
from privacypacking.schedulers.threshold_updating import ThresholdUpdating
from privacypacking.schedulers.utils import (
    BASIC_SCHEDULER,
    BUDGET_UNLOCKING,
    THRESHOLD_UPDATING,
    SIMPLEX,
)

# Loading them so that they get stored in globals()
from privacypacking.schedulers.metrics import (
    dominant_shares,
    fcfs,
    flat_relevance,
    online_flat_relevance,
    overflow_relevance,
    round_robins,
)


def get_scheduler(config) -> Scheduler:
    schedulers = {
        BASIC_SCHEDULER: Scheduler,
        BUDGET_UNLOCKING: NBudgetUnlocking,
        THRESHOLD_UPDATING: ThresholdUpdating,
        SIMPLEX: simplex.Simplex,
    }
    if config.scheduler_method == SIMPLEX:
        return schedulers[config.scheduler_method]()
    else:
        metric = None
        if config.scheduler_metric in globals():
            metric = globals()[config.scheduler_metric]
        assert metric is not None

        # Some schedulers might need custom arguments
        if config.scheduler_method == BASIC_SCHEDULER:
            return schedulers[config.scheduler_method](metric)
        elif config.scheduler_method == BUDGET_UNLOCKING:
            return schedulers[config.scheduler_method](metric, config.scheduler_N)
        elif config.scheduler_method == THRESHOLD_UPDATING:
            scheduler_threshold_update_mechanism = ThresholdUpdateMechanism.from_str(
                config.scheduler_threshold_update_mechanism
            )
            return schedulers[config.scheduler_method](
                metric,
                scheduler_threshold_update_mechanism,
            )
        else:
            exit()
