import os
import numpy as np
import argparse
from loguru import logger
from ray import tune
import yaml

from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import *
from privacypacking.schedulers.utils import (
    SIMPLEX,
)
from privacypacking.config import Config


def run_and_report(config: dict) -> None:
    os.environ["LOGURU_LEVEL"] = "INFO"
    sim = Simulator(Config(config))
    metrics = sim.run()
    logger.info(metrics)
    tune.report(**metrics)


def grid():
    scheduler_methods = [SIMPLEX]
    block_selection_policies = ["RandomBlocks"]
    # num_blocks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

    config[SCHEDULER_SPEC][METHOD] = tune.grid_search(scheduler_methods)
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][INITIAL_NUM] = tune.grid_search(
        np.arange(0, 1100, step=100, dtype=int).tolist()
    )
    # config[BLOCKS_SPEC][INITIAL_NUM] = tune.grid_search(num_blocks)

    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
        READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    ][BLOCK_SELECTING_POLICY] = tune.grid_search(block_selection_policies)

    tune.run(
        run_and_report,
        config=config,
        resources_per_trial={"cpu": 32},
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

    grid()