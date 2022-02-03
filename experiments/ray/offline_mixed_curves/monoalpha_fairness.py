import argparse
import os
from datetime import datetime

import loguru
from loguru import logger
from ray import tune
from ray.tune import Stopper
from ray.tune.stopper import TimeoutStopper

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
    SQUARED_DYNAMIC_FLAT_RELEVANCE,
    TASK_BASED_BUDGET_UNLOCKING,
    TESSERACTED_DYNAMIC_FLAT_RELEVANCE,
    THRESHOLD_UPDATING,
    TIME_BASED_BUDGET_UNLOCKING,
)
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import *


class TrialStopper(Stopper):
    def __init__(self, max_seconds=10):
        self._deadline = max_seconds

    def __call__(self, trial_id, result):
        logger.warning(
            f"Trial {trial_id} has been running for {result['time_total_s']} seconds"
        )
        return result["time_total_s"] > self._deadline

    def stop_all(self):
        return False


def run_and_report(config: dict) -> None:
    # Unpack conditional parameters
    config[SCHEDULER_SPEC][METHOD], config[SCHEDULER_SPEC][METRIC] = config.pop(
        "method_and_metric"
    )

    logger.info(f"Running simulator with config: {config}")

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

    temperature = [1]
    normalize_by = ["available_budget"]
    metric_recomputation_period = [10]

    # Conditonal parameter
    method_and_metric = []
    for metric in [
        DOMINANT_SHARES,
        # FLAT_RELEVANCE,
        # # OVERFLOW_RELEVANCE,
        # FCFS,
        # SOFT_KNAPSACK,
        # DYNAMIC_FLAT_RELEVANCE,
    ]:
        method_and_metric.append((BASIC_SCHEDULER, metric))
    # method_and_metric.append((SIMPLEX, DOMINANT_SHARES))

    config["method_and_metric"] = tune.grid_search(method_and_metric)

    # num_tasks = [50, 100, 200, 300, 350, 400, 500, 750, 1000, 1500, 2000]
    num_tasks = [500]

    num_blocks = [20]
    data_path = "mixed_curves"
    block_selection_policies = ["RandomBlocks"]

    config[BLOCKS_SPEC][INITIAL_NUM] = tune.grid_search(num_blocks)
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM].update(
        {
            SAMPLING: True,
            INITIAL_NUM: tune.grid_search(num_tasks),
            DATA_PATH: data_path,
            DATA_TASK_FREQUENCIES_PATH: "frequencies.yaml",
            FREQUENCY: 1,
            READ_BLOCK_SELECTION_POLICY_FROM_CONFIG: {
                ENABLED: True,
                BLOCK_SELECTING_POLICY: tune.grid_search(block_selection_policies),
            },
        }
    )

    # config[BLOCKS_SPEC][INITIAL_NUM] = tune.grid_search(num_blocks)
    # # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][SAMPLING] = True
    # # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][INITIAL_NUM] = tune.grid_search(
    # #     num_tasks
    # # )
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][INITIAL_NUM] = tune.grid_search(
    #     num_tasks
    # )
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
    #     READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    # ][ENABLED] = True
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
    #     READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    # ][BLOCK_SELECTING_POLICY] = tune.grid_search(block_selection_policies)
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][DATA_PATH] = data_path
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
    #     DATA_TASK_FREQUENCIES_PATH
    # ] = "frequencies.yaml"

    config[CUSTOM_LOG_PREFIX] = f"exp_{datetime.now().strftime('%m%d-%H%M%S')}"

    config["omegaconf"] = {
        "scheduler": {
            "metric_recomputation_period": tune.grid_search(
                metric_recomputation_period
            ),
            "log_warning_every_n_allocated_tasks": 0,
        },
        "metric": {
            "normalize_by": tune.grid_search(normalize_by),
            "temperature": tune.grid_search(temperature),
            "gurobi_timeout": 1_000,
        },
        "logs": {
            "verbose": False,
            "save": True,
        },
        "alphas": [4.0],
    }

    tune.run(
        run_and_report,
        config=config,
        resources_per_trial={"cpu": 1},
        local_dir=RAY_LOGS,
        resume=False,
        # stop=TrialStopper(max_seconds=30),
        # stop=TimeoutStopper(timeout=3 * 60),
    )


if __name__ == "__main__":
    os.environ["LOGURU_LEVEL"] = "WARNING"
    grid()
