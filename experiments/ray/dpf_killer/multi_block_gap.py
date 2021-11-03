import argparse
import os
from datetime import datetime

import numpy as np
from loguru import logger
from ray import tune

from privacypacking.config import Config
from privacypacking.schedulers.utils import BASIC_SCHEDULER, DOMINANT_SHARES, SIMPLEX
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
            "offline_dpf_killer/multi_block/gap_base.yaml"
        ),
        "r",
    ) as user_config:
        user_config = yaml.safe_load(user_config)
    update_dict(user_config, config)

    scheduler_methods = [BASIC_SCHEDULER, SIMPLEX]
    scheduler_metrics = [DOMINANT_SHARES]
    # block_selection_policies = ["RandomBlocks", "Pareto_1"]
    block_selection_policies = [
        "RandomBlocks",
        "Zeta_1",
        "Zeta_0.5",
        "ContiguousBlocksRandomOffset",
    ]

    num_tasks = [100]
    num_blocks = [5, 10, 15, 20]
    # num_blocks = [5, 10]

    config[SCHEDULER_SPEC][METHOD] = tune.grid_search(scheduler_methods)
    config[SCHEDULER_SPEC][METRIC] = tune.grid_search(scheduler_metrics)
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][INITIAL_NUM] = tune.grid_search(
    #     np.arange(1, 500, step=1, dtype=int).tolist()
    # )
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][INITIAL_NUM] = tune.grid_search(
        num_tasks
    )
    config[BLOCKS_SPEC][INITIAL_NUM] = tune.grid_search(num_blocks)

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

    grid()
