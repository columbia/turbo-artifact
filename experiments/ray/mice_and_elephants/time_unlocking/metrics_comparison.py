import os

import argparse
from loguru import logger
from ray import tune

from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import *
from privacypacking.schedulers.utils import (
    BASIC_SCHEDULER,
    TIME_BASED_BUDGET_UNLOCKING,
    TASK_BASED_BUDGET_UNLOCKING,
    SIMPLEX,
    THRESHOLD_UPDATING,
    DOMINANT_SHARES,
    FLAT_RELEVANCE,
    DYNAMIC_FLAT_RELEVANCE,
    SQUARED_DYNAMIC_FLAT_RELEVANCE,
    TESSERACTED_DYNAMIC_FLAT_RELEVANCE,
    OVERFLOW_RELEVANCE,
    FCFS,
    NAIVE_AVERAGE,
)
from privacypacking.config import Config


def run_and_report(config: dict) -> None:
    os.environ["LOGURU_LEVEL"] = "INFO"
    sim = Simulator(Config(config))
    metrics = sim.run()
    logger.info(metrics)
    tune.report(**metrics)


def grid():
    scheduler_methods = [TIME_BASED_BUDGET_UNLOCKING]
    scheduler_metrics = [
        DOMINANT_SHARES,
        FLAT_RELEVANCE,
        DYNAMIC_FLAT_RELEVANCE,
        SQUARED_DYNAMIC_FLAT_RELEVANCE,
        TESSERACTED_DYNAMIC_FLAT_RELEVANCE,
    ]
    scheduling_wait_time = [0.25]
    budget_unlocking_time = [0.025]
    n = [125, 500, 1000]

    # threshold_update_mechanisms = [NAIVE_AVERAGE]
    data_task_frequencies_path = ["mice_0.yaml", "mice_20.yaml", "mice_40.yaml", "mice_60.yaml", "mice_80.yaml",
                                  "mice_100.yaml"]
    config[SCHEDULING_WAIT_TIME] = tune.grid_search(scheduling_wait_time)
    config[BUDGET_UNLOCKING_TIME] = tune.grid_search(budget_unlocking_time)
    config[SCHEDULER_SPEC][METHOD] = tune.grid_search(scheduler_methods)
    config[SCHEDULER_SPEC][METRIC] = tune.grid_search(scheduler_metrics)
    config[SCHEDULER_SPEC][N] = tune.grid_search(n)

    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
        DATA_TASK_FREQUENCIES_PATH
    ] = tune.grid_search(data_task_frequencies_path)
    # config[THRESHOLD_UPDATE_MECHANISM] = tune.grid_search(threshold_update_mechanisms)
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

    grid()
