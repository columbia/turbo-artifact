import argparse
import os
from datetime import datetime

import scalene
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


def id(x):
    return x[0]


def run_one_config():
    scheduler_methods = [TIME_BASED_BUDGET_UNLOCKING]
    scheduler_metrics = [
        VECTORIZED_BATCH_OVERFLOW_RELEVANCE,
        # BATCH_OVERFLOW_RELEVANCE,
        # DOMINANT_SHARES,
        # FLAT_RELEVANCE,
        # DYNAMIC_FLAT_RELEVANCE,
        # FCFS,
        # SOFTMAX_OVERFLOW,
    ]

    # TODO: add temperature to config
    # scheduler_scheduling_time = [0.01, 0.1, 0.5, 1, 2, 5, 10, 15, 20, 25]
    # scheduler_scheduling_time = [0.01, 0.5, 1, 5, 10, 20]
    scheduler_scheduling_time = [1]
    # scheduler_scheduling_time = [0.1, 1, 10]

    # n = [100, 500, 1000, 1500, 2000]

    n = [10_000]
    data_lifetime = [5]

    # n = [1]
    # data_lifetime = [0.001]

    avg_number_tasks_per_block = [100]
    # avg_number_tasks_per_block = [50]
    # avg_number_tasks_per_block = [10, 25, 50]

    max_blocks = [20]
    # max_blocks = [60]

    # TODO: re-add the initial blocks
    initial_blocks = [0]
    seeds = [0]

    # TODO: rescale (more tasks?) to separate batch OR and dyn FR

    # data_path = "privatekube_event_g0.3_l0.3_p=1"
    data_path = "mixed_curves"
    # data_path = "mixed_curves_killer"

    # data_path = "mixed_curves_large"

    block_selection_policies = ["LatestBlocksFirst"]

    config[GLOBAL_SEED] = id(seeds)
    config[BLOCKS_SPEC][INITIAL_NUM] = id(initial_blocks)

    # config[TASKS_SPEC][MAX_TASKS][FROM_MAX_BLOCKS] = id(max_blocks)
    config[BLOCKS_SPEC][MAX_BLOCKS] = id(max_blocks)

    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
        READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    ][ENABLED] = True
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
        READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    ][BLOCK_SELECTING_POLICY] = id(block_selection_policies)
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][DATA_PATH] = data_path

    config[TASKS_SPEC][TASK_ARRIVAL_FREQUENCY][POISSON][
        AVG_NUMBER_TASKS_PER_BLOCK
    ] = id(avg_number_tasks_per_block)

    config[SCHEDULER_SPEC][DATA_LIFETIME] = id(data_lifetime)
    config[SCHEDULER_SPEC][SCHEDULING_WAIT_TIME] = id(scheduler_scheduling_time)
    config[SCHEDULER_SPEC][METHOD] = id(scheduler_methods)
    config[SCHEDULER_SPEC][METRIC] = id(scheduler_metrics)
    config[SCHEDULER_SPEC][N] = id(n)
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][INITIAL_NUM] = id(np.arange(0, 5100, step=100, dtype=int).tolist())
    config[CUSTOM_LOG_PREFIX] = f"exp_{datetime.now().strftime('%m%d-%H%M%S')}"

    logger.info(f"Tune config: {config}")

    sim = Simulator(Config(config))
    metrics = sim.run()
    logger.info(metrics)


if __name__ == "__main__":

    with open(DEFAULT_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    with open(
        "privacypacking/config/time_based_budget_unlocking/privatekube/base.yaml", "r"
    ) as user_config:
        user_config = yaml.safe_load(user_config)
    update_dict(user_config, config)
    os.environ["LOGURU_LEVEL"] = "WARNING"
    # os.environ["LOGURU_LEVEL"] = "INFO"

    # Turn profiling on
    # scalene_profiler.start()

    run_one_config()

    # Turn profiling off
    # scalene_profiler.stop()
