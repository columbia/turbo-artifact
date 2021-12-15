import argparse
import os
from datetime import datetime

import ray
from loguru import logger
from ray import tune

from privacypacking import schedulers
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
    SOFT_KNAPSACK,
    SOFTMAX_OVERFLOW,
    SQUARED_DYNAMIC_FLAT_RELEVANCE,
    TASK_BASED_BUDGET_UNLOCKING,
    TESSERACTED_DYNAMIC_FLAT_RELEVANCE,
    THRESHOLD_UPDATING,
    TIME_BASED_BUDGET_UNLOCKING,
    VECTORIZED_BATCH_OVERFLOW_RELEVANCE,
)
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import *


def run_and_report(config: dict) -> None:
    sim = Simulator(Config(config))
    metrics = sim.run()
    logger.info(metrics)
    tune.report(**metrics)


def grid():

    # ray.init(log_to_driver=False)

    scheduler_methods = [TIME_BASED_BUDGET_UNLOCKING]
    scheduler_metrics = [
       SOFT_KNAPSACK,
         BATCH_OVERFLOW_RELEVANCE,
         FLAT_RELEVANCE,
         DYNAMIC_FLAT_RELEVANCE,
        #  FCFS,
        # # VECTORIZED_BATCH_OVERFLOW_RELEVANCE,
         DOMINANT_SHARES,
    ]

    # temperature = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3, 4, 5]
    # temperature = [0.001, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000]
    temperature = [1]

    # normalize_by = ["capacity", "available_budget", ""]
    # normalize_by = [""]
    normalize_by = ["available_budget"]

    metric_recomputation_period = [10]

    # Fully unlocked case
    n = [1]
    data_lifetime = [0.001]

    scheduler_scheduling_time = [1]

#    avg_number_tasks_per_block = [100] 
    avg_number_tasks_per_block = [50, 100, 200, 300, 400, 500]
    max_blocks = [20]
    initial_blocks = [19]
    seeds = [1]
    block_selection_policies = ["RandomBlocks"]

    data_path = "mixed_curves"
    # data_path = "mixed_curves_profits_0.5-1.0"

    config[GLOBAL_SEED] = tune.grid_search(seeds)
    config[BLOCKS_SPEC][INITIAL_NUM] = tune.grid_search(initial_blocks)
    config[BLOCKS_SPEC][MAX_BLOCKS] = tune.grid_search(max_blocks)

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

    config[SCHEDULER_SPEC][DATA_LIFETIME] = tune.grid_search(data_lifetime)
    config[SCHEDULER_SPEC][SCHEDULING_WAIT_TIME] = tune.grid_search(
        scheduler_scheduling_time
    )
    config[SCHEDULER_SPEC][METHOD] = tune.grid_search(scheduler_methods)
    config[SCHEDULER_SPEC][METRIC] = tune.grid_search(scheduler_metrics)
    config[SCHEDULER_SPEC][N] = tune.grid_search(n)
    config[CUSTOM_LOG_PREFIX] = f"exp_{datetime.now().strftime('%m%d-%H%M%S')}"

    config["omegaconf"] = {
        "scheduler": {
            "metric_recomputation_period": tune.grid_search(
                metric_recomputation_period
            ),
        },
        "metric": {
            "normalize_by": tune.grid_search(normalize_by),
            "temperature": tune.grid_search(temperature),
        },
    }

    logger.info(f"Tune config: {config}")

    tune.run(
        run_and_report,
        config=config,
        resources_per_trial={"cpu": 1},
        # resources_per_trial={"cpu": 32},
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
    os.environ["LOGURU_LEVEL"] = "WARNING"
    # os.environ["LOGURU_LEVEL"] = "INFO"

    grid()
