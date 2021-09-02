"""Script to launch sequences of experiments.
Runs in parallel with Ray and gathers the hyperparameters and results in a TensorBoard.

Usage: modify this script with the configuration logic you need.
"""

import os

import yaml
import argparse
from loguru import logger
from ray import tune
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import *
from privacypacking.config import Config


def run_and_report(config: dict) -> None:
    os.environ["LOGURU_LEVEL"] = "INFO"
    metrics = Simulator(Config(config)).run()
    logger.info(metrics)
    tune.report(**metrics)


def custom_search_space():
    global config
    global schedulers
    # A list of configuration parameters that override the default parameters
    search_space = []

    for scheduler_name in schedulers:
        # Everything identical except the scheduler
        extra_config = {"scheduler_spec": {"name": scheduler_name}}
        search_space.append(extra_config)

    analysis = tune.run(
        run_and_report,
        config=config,
        resources_per_trial={"cpu": 1},
        local_dir=RAY_LOGS,
        resume=False,
        search_alg=BasicVariantGenerator(points_to_evaluate=search_space),
        num_samples=len(search_space),
    )


def grid():
    global config
    config["blocks_spec"]["initial_num"] = tune.grid_search([5, 10])

    for curve in ["laplace", "gaussian", "SubsampledGaussian"]:
        config["tasks_spec"]["curve_distributions"][curve][
            "initial_num"
        ] = tune.grid_search([50])

    config["scheduler_spec"]["name"] = tune.grid_search(schedulers)

    tune.run(
        run_and_report,
        config=config,
        resources_per_trial={"cpu": 1},
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

    schedulers = [
        "dpf",
        "fcfs",
        # "FlatRelevance",
        # "OverflowRelevance",
        "simplex",
    ]
    grid()
