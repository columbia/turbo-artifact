import argparse
import os

from loguru import logger
from ray import tune

from privacypacking.config import Config
from privacypacking.schedulers.utils import (
    BASIC_SCHEDULER,
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
    ]
    scheduler_budget_unlocking_time = [0.025]
    # scheduler_scheduling_time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    scheduler_scheduling_time = [0.25]
    # n = [100, 500, 1000, 1500, 2000]
    n = [200]

    block_selection_policies = ["LatestBlocksFirst"]

    config[TASKS_SPEC][MAX_TASKS][NUM] = 100

    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
        READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    ][ENABLED] = True
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
        READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    ][BLOCK_SELECTING_POLICY] = tune.grid_search(block_selection_policies)
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][DATA_PATH] = 

    config[BUDGET_UNLOCKING_TIME] = tune.grid_search(scheduler_budget_unlocking_time)
    config[SCHEDULING_WAIT_TIME] = tune.grid_search(scheduler_scheduling_time)
    config[SCHEDULER_SPEC][METHOD] = tune.grid_search(scheduler_methods)
    config[SCHEDULER_SPEC][METRIC] = tune.grid_search(scheduler_metrics)
    config[SCHEDULER_SPEC][N] = tune.grid_search(n)
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][INITIAL_NUM] = tune.grid_search(np.arange(0, 5100, step=100, dtype=int).tolist())

    tune.run(
        run_and_report,
        config=config,
        # resources_per_trial={"cpu": 1},
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
