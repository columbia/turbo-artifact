import argparse
import os
from datetime import datetime
from functools import partial
from typing import Any, Dict, List

import ray
import yaml
from loguru import logger
from ray import tune

from privacypacking import schedulers
from privacypacking.config import Config
from privacypacking.schedulers.utils import (
    ARGMAX_KNAPSACK,
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
    THRESHOLD_UPDATING,
    TIME_BASED_BUDGET_UNLOCKING,
    VECTORIZED_BATCH_OVERFLOW_RELEVANCE,
)
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import *


def run_and_report(config: dict, packed=False) -> None:

    if packed:
        # Unpack conditional parameters
        config[SCHEDULER_SPEC][METHOD], config[SCHEDULER_SPEC][METRIC] = config.pop(
            "method_and_metric"
        )

    sim = Simulator(Config(config))
    metrics = sim.run()
    # logger.info(f"Trial logs: {tune.get_trial_dir()}")
    tune.report(**metrics)

    # return tune.get_trial_dir()


def grid_offline(
    custom_config: str,
    num_tasks: List[int],
    num_blocks: List[int],
):
    with open(DEFAULT_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    with open(
        DEFAULT_CONFIG_FILE.parent.joinpath(custom_config),
        "r",
    ) as user_config:
        user_config = yaml.safe_load(user_config)
    update_dict(user_config, config)

    # Conditonal parameter
    method_and_metric = []
    for metric in [
        DOMINANT_SHARES,
        FLAT_RELEVANCE,
        OVERFLOW_RELEVANCE,
        ARGMAX_KNAPSACK,
    ]:
        method_and_metric.append((BASIC_SCHEDULER, metric))

    method_and_metric.append((SIMPLEX, DOMINANT_SHARES))

    config["method_and_metric"] = tune.grid_search(method_and_metric)

    block_selection_policies = [
        "RandomBlocks",
    ]

    # num_tasks = [50, 100, 150, 200]
    # num_tasks = [100]
    # num_blocks = [5, 10, 15, 20]
    # num_blocks = [5]

    temperature = [-1]

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

    config["omegaconf"] = {
        "scheduler": {
            "metric_recomputation_period": 10,
            "log_warning_every_n_allocated_tasks": 500,
            "scheduler_timeout_seconds": 20 * 60,
        },
        "metric": {
            "normalize_by": "available_budget",
            "temperature": tune.grid_search(temperature),
            "n_knapsack_solvers": 16,
            "gurobi_timeout": 10 * 60,
            "gurobi_threads": 8,
        },
        "logs": {
            "verbose": False,
            "save": True,
        },
    }

    experiment_analysis = tune.run(
        partial(run_and_report, packed=True),
        config=config,
        resources_per_trial={"cpu": 1},
        local_dir=RAY_LOGS,
        resume=False,
        verbose=0,
        callbacks=[
            CustomLoggerCallback(),
            tune.logger.JsonLoggerCallback(),
            # tune.integration.mlflow.MLflowLoggerCallback(
            #     experiment_name="grid_offline",
            # ),
        ],
    )

    return experiment_analysis


def grid_online(
    custom_config: str = "time_based_budget_unlocking/privatekube/base.yaml",
):
    with open(DEFAULT_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    with open(
        DEFAULT_CONFIG_FILE.parent.joinpath(custom_config),
        "r",
    ) as user_config:
        user_config = yaml.safe_load(user_config)
    update_dict(user_config, config)
    # ray.init(log_to_driver=False)

    scheduler_methods = [TIME_BASED_BUDGET_UNLOCKING]
    scheduler_metrics = [
        SOFT_KNAPSACK,
        # ARGMAX_KNAPSACK,
        # BATCH_OVERFLOW_RELEVANCE,
        #  FLAT_RELEVANCE,
        # DYNAMIC_FLAT_RELEVANCE,
        #  FCFS,
        # # VECTORIZED_BATCH_OVERFLOW_RELEVANCE,
        # DOMINANT_SHARES,
    ]

    # temperature = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3, 4, 5]
    temperature = [0.001, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000]
    # temperature = [0.01]

    # normalize_by = ["capacity", "available_budget", ""]
    # normalize_by = [""]
    normalize_by = ["available_budget"]

    metric_recomputation_period = [500]

    # Fully unlocked case
    # n = [1]
    # data_lifetime = [0.001]

    # Progressive unlocking
    n = [1_000]
    data_lifetime = [5]

    # scheduler_scheduling_time = [0.01, 0.1, 0.5, 1.0, 2.0, 4, 6, 8, 10, 20, 30]
    # scheduler_scheduling_time = [0.1, 1.0, 4, 8, 20]
    scheduler_scheduling_time = [1]

    avg_number_tasks_per_block = [500]
    # avg_number_tasks_per_block = [500]
    max_blocks = [30]
    initial_blocks = [10]
    seeds = [0]
    block_selection_policies = ["LatestBlocksFirst"]

    # data_path = "mixed_curves"
    data_path = "mixed_curves_profits"

    config[GLOBAL_SEED] = tune.grid_search(seeds)
    config[BLOCKS_SPEC][INITIAL_NUM] = tune.grid_search(initial_blocks)
    config[BLOCKS_SPEC][MAX_BLOCKS] = tune.grid_search(max_blocks)

    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
        READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    ][ENABLED] = True
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
        READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    ][BLOCK_SELECTING_POLICY] = tune.grid_search(block_selection_policies)
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][DATA_PATH] = data_path

    config[TASKS_SPEC][TASK_ARRIVAL_FREQUENCY][POISSON][
        AVG_NUMBER_TASKS_PER_BLOCK
    ] = tune.grid_search(avg_number_tasks_per_block)

    config[SCHEDULER_SPEC][DATA_LIFETIME] = tune.grid_search(data_lifetime)
    config[SCHEDULER_SPEC][SCHEDULING_WAIT_TIME] = tune.grid_search(
        scheduler_scheduling_time
    )
    config[SCHEDULER_SPEC][METHOD] = tune.grid_search(scheduler_methods)
    config[SCHEDULER_SPEC][METRIC] = tune.grid_search(scheduler_metrics)
    config[SCHEDULER_SPEC][N] = tune.grid_search(n)
    config[CUSTOM_LOG_PREFIX] = f"exp_{datetime.now().strftime('%m%d-%H%M%S')}"

    config["omegaconf"] = {
        "scheduler": {
            "metric_recomputation_period": tune.grid_search(
                metric_recomputation_period
            ),
            "log_warning_every_n_allocated_tasks": 500,
            "scheduler_timeout_seconds": 20 * 60,
        },
        "metric": {
            "normalize_by": tune.grid_search(normalize_by),
            "temperature": tune.grid_search(temperature),
            "n_knapsack_solvers": 16,
        },
        "logs": {
            "verbose": False,
            "save": True,
        },
    }

    logger.info(f"Tune config: {config}")

    tune.run(
        run_and_report,
        config=config,
        resources_per_trial={"cpu": 1},
        # resources_per_trial={"cpu": 32},
        local_dir=RAY_LOGS,
        resume=False,
        verbose=0,
        callbacks=[
            CustomLoggerCallback(),
            tune.logger.JsonLoggerCallback(),
            tune.integration.mlflow.MLflowLoggerCallback(
                experiment_name="mixed_curves_online",
            ),
        ],
        progress_reporter=ray.tune.CLIReporter(
            metric_columns=["n_allocated_tasks", "total_tasks", "realized_profit"],
            parameter_columns={
                "scheduler_spec/scheduling_wait_time": "T",
                "scheduler_spec/data_lifetime": "lifetime",
                "scheduler_spec/metric": "metric",
                "omegaconf/metric/temperature": "temperature",
            },
            max_report_frequency=60,
        ),
    )


class CustomLoggerCallback(tune.logger.LoggerCallback):
    """Custom logger interface"""

    def __init__(self) -> None:
        super().__init__()

    def log_trial_result(self, iteration: int, trial: Any, result: Dict):
        logger.info(
            [
                f"{key}: {result[key]}"
                for key in ["n_allocated_tasks", "realized_profit", "temperature"]
            ]
        )
        return

    def on_trial_complete(self, iteration: int, trials: List, trial: Any, **info):
        return
