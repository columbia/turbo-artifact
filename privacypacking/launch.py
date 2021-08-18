"""Script to launch sequences of experiments.
Runs in parallel with Ray and gathers the hyperparameters and results in a TensorBoard.

Usage: modify this script with the configuration logic you need.
"""

import os

import yaml
from loguru import logger
from ray import tune
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from privacypacking.discrete_simulator import run
from privacypacking.utils.utils import *


def run_and_report(config: dict) -> None:
    os.environ["LOGURU_LEVEL"] = "INFO"
    metrics = run(config)
    logger.info(metrics)
    tune.report(**metrics)


def custom_search_space():

    with open(DEFAULT_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    with open(
        DEFAULT_CONFIG_FILE.parent.joinpath("offline_dpf.yaml"), "r"
    ) as user_config:
        user_config = yaml.safe_load(user_config)

    update_dict(user_config, config)

    # A list of configuration parameters that override the default parameters
    search_space = []

    for scheduler_name in [
        "OfflineDPF",
        "FlatRelevance",
        "OverflowRelevance",
        "simplex",
    ]:
        # Everything identical except the scheduler
        extra_config = {"scheduler_spec": {"name": scheduler_name}}
        search_space.append(extra_config)

    # (I had to apply this patch: https://github.com/ray-project/ray/pull/17282/files)
    # Comment out the warning in lib/python3.9/site-packages/ray/tune/suggest/variant_generator.py"
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
    with open(DEFAULT_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    with open(
        DEFAULT_CONFIG_FILE.parent.joinpath("offline_dpf.yaml"), "r"
    ) as user_config:
        user_config = yaml.safe_load(user_config)

    update_dict(user_config, config)

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
