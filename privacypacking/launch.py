"""Script to launch sequences of experiments.
Runs in parallel with Ray and gathers the hyperparameters and results in a TensorBoard.

Usage: modify this script with the configuration logic you need.
"""

import os

import numpy as np
import yaml
from loguru import logger
from ray import tune
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from privacypacking.config import Config
from privacypacking.discrete_simulator import run
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import *


def run_and_report(config: dict) -> None:
    os.environ["LOGURU_LEVEL"] = "INFO"
    metrics = run(config)
    # metrics = Simulator(Config(config)).run()
    logger.info(metrics)
    tune.report(**metrics)


def grid():
    with open(DEFAULT_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    with open(
        DEFAULT_CONFIG_FILE.parent.joinpath("offline_multiblock_dpf_killer.yaml"), "r"
    ) as user_config:
        user_config = yaml.safe_load(user_config)
    update_dict(user_config, config)

    config["tasks_spec"]["initial_num"] = tune.grid_search(
        np.arange(1, 100, step=5, dtype=int).tolist()
    )

    # config["blocks_spec"]["initial_num"] = tune.grid_search([5, 10])

    # for curve in ["laplace", "gaussian", "SubsampledGaussian"]:
    #     config["tasks_spec"]["curve_distributions"][curve][
    #         "initial_num"
    #     ] = tune.grid_search([0, 50, 85, 100, 175])
    config["scheduler_spec"]["name"] = tune.grid_search(
        [
            "OfflineDPF",
            "FlatRelevance",
            "OverflowRelevance",
            "simplex",
        ]
    )

    tune.run(
        run_and_report,
        config=config,
        resources_per_trial={"cpu": 1},
        local_dir=RAY_LOGS,
        resume=False,
    )


if __name__ == "__main__":
    grid()
