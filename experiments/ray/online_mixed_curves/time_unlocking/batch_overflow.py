import argparse
import os
from datetime import datetime

from loguru import logger
from ray import tune

from privacypacking.config import Config
from privacypacking.schedulers.utils import (
    BASIC_SCHEDULER,
    BATCH_OVERFLOW_RELEVANCE,
    DOMINANT_SHARES,
    DYNAMIC_FLAT_RELEVANCE,
    FCFS,
    FLAT_RELEVANCE,
    NAIVE_AVERAGE,
    OVERFLOW_RELEVANCE,
    SIMPLEX,
    SQUARED_DYNAMIC_FLAT_RELEVANCE,
    TASK_BASED_BUDGET_UNLOCKING,
    TESSERACTED_DYNAMIC_FLAT_RELEVANCE,
    THRESHOLD_UPDATING,
    TIME_BASED_BUDGET_UNLOCKING,
)
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import *


def run_and_report(config: dict) -> None:
    sim = Simulator(Config(config))
    metrics = sim.run()
    logger.info(metrics)
    tune.report(**metrics)


def grid():
    scheduler_methods = [TIME_BASED_BUDGET_UNLOCKING]
    scheduler_metrics = [
        BATCH_OVERFLOW_RELEVANCE,
        DOMINANT_SHARES,
        FLAT_RELEVANCE,
        DYNAMIC_FLAT_RELEVANCE,
        FCFS,
    ]

    scheduler_scheduling_time = [0.1]
    # n = [100, 500, 1000, 1500, 2000]
    n = [1]
    data_lifetime = [1]
    avg_number_tasks_per_block = [10]
    max_blocks = [10]
    initial_blocks = [5]

    # TODO: rescale (more tasks?) to separate batch OR and dyn FR

    # data_path = "privatekube_event_g0.3_l0.3_p=1"
    # data_path = "mixed_curves"
    data_path = "mixed_curves_large"

    # TODO: warm up and wind down period?

    # TODO: Try to create initial blocks already unlocked?

    block_selection_policies = ["LatestBlocksFirst"]

    config[BLOCKS_SPEC][INITIAL_NUM] = tune.grid_search(initial_blocks)

    config[TASKS_SPEC][MAX_TASKS][FROM_MAX_BLOCKS] = tune.grid_search(max_blocks)

    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
        READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    ][ENABLED] = True
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
        READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    ][BLOCK_SELECTING_POLICY] = tune.grid_search(block_selection_policies)
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][DATA_PATH] = data_path

    config[TASKS_SPEC][TASK_ARRIVAL_FREQUENCY][POISSON][
        AVG_NUMBER_TASKS_PER_BLOCK
    ] = tune.grid_search(avg_number_tasks_per_block)

    config[DATA_LIFETIME] = tune.grid_search(data_lifetime)
    config[SCHEDULING_WAIT_TIME] = tune.grid_search(scheduler_scheduling_time)
    config[SCHEDULER_SPEC][METHOD] = tune.grid_search(scheduler_methods)
    config[SCHEDULER_SPEC][METRIC] = tune.grid_search(scheduler_metrics)
    config[SCHEDULER_SPEC][N] = tune.grid_search(n)
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][INITIAL_NUM] = tune.grid_search(np.arange(0, 5100, step=100, dtype=int).tolist())
    config[CUSTOM_LOG_PREFIX] = f"exp_{datetime.now().strftime('%m%d-%H%M%S')}"

    tune.run(
        run_and_report,
        config=config,
        resources_per_trial={"cpu": 3},
        local_dir=RAY_LOGS,
        resume=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_file")
    args = parser.parse_args()
    with open(DEFAULT_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    with open(args.config_file, "r") as user_config:
        user_config = yaml.safe_load(user_config)
    update_dict(user_config, config)
    os.environ["LOGURU_LEVEL"] = "INFO"

    grid()
