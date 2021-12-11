import argparse
import os
from datetime import datetime

import numpy as np
from loguru import logger
from ray import tune

from privacypacking.config import Config
from privacypacking.schedulers.utils import (
    BASIC_SCHEDULER,
    DOMINANT_SHARES,
    FLAT_RELEVANCE,
    OVERFLOW_RELEVANCE,
    SIMPLEX,
    SOFT_KNAPSACK,
)
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import *


def run_and_report(config: dict) -> None:
    os.environ["LOGURU_LEVEL"] = "INFO"
    sim = Simulator(Config(config))
    metrics = sim.run()
    logger.info(metrics)
    tune.report(**metrics)


def grid():

    with open(DEFAULT_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    with open(
        DEFAULT_CONFIG_FILE.parent.joinpath(
            "offline_dpf_killer/single_block/base.yaml"
        ),
        "r",
    ) as user_config:
        user_config = yaml.safe_load(user_config)
    update_dict(user_config, config)

    scheduler_methods = [BASIC_SCHEDULER, SIMPLEX]
    scheduler_metrics = [
        DOMINANT_SHARES,
        FLAT_RELEVANCE,
        OVERFLOW_RELEVANCE,
        SOFT_KNAPSACK,
    ]
    block_selection_policies = ["RandomBlocks"]

    config[SCHEDULER_SPEC][METHOD] = tune.grid_search(scheduler_methods)
    config[SCHEDULER_SPEC][METRIC] = tune.grid_search(scheduler_metrics)
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][INITIAL_NUM] = tune.grid_search(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 200, 300, 400, 500]
    )
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
        READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    ][BLOCK_SELECTING_POLICY] = tune.grid_search(block_selection_policies)

    config[CUSTOM_LOG_PREFIX] = f"exp_{datetime.now().strftime('%m%d-%H%M%S')}"

    tune.run(
        run_and_report,
        config=config,
        resources_per_trial={"cpu": 1},
        local_dir=RAY_LOGS,
        resume=False,
    )


if __name__ == "__main__":
    os.environ["LOGURU_LEVEL"] = "WARNING"
    grid()
