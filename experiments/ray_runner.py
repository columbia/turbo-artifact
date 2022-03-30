import argparse
import os
from datetime import datetime
from functools import partial
from typing import Any, Dict, List

import ray
import yaml
from loguru import logger
from ray import tune

from experiments.ray.analysis import load_ray_experiment
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
from privacypacking.utils.generate_curves import P_GRID
from privacypacking.utils.utils import *


def run_and_report(config: dict) -> None:
    sim = Simulator(Config(config))
    metrics = sim.run()
    # logger.info(f"Trial logs: {tune.get_trial_dir()}")
    tune.report(**metrics)

    # return tune.get_trial_dir()


def grid_offline(
    num_tasks: List[int],
    num_blocks: List[int],
    data_path: str = "",
    optimal: bool = False,
    metric_recomputation_period: int = 10,
    parallel: bool = False,
    gurobi_timeout_minutes: int = 1,
):
    # TODO: remove the remaining stuff in there
    with open(DEFAULT_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    metrics = [
        SIMPLEX,
        DOMINANT_SHARES,
        FLAT_RELEVANCE,
        OVERFLOW_RELEVANCE,
        ARGMAX_KNAPSACK,
    ]

    num_blocks = tune.grid_search(num_blocks)
    block_selection_policies = ["RandomBlocks"]
    temperature = [-1]

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

    n_knapsack_solvers = os.cpu_count() // 8 if parallel else 1
    gurobi_threads = os.cpu_count() // 4

    config["omegaconf"] = {
        "scheduler": {
            "method": "offline",
            "metric": tune.grid_search(metrics),
            "metric_recomputation_period": metric_recomputation_period,
            "log_warning_every_n_allocated_tasks": 50,
            "scheduler_timeout_seconds": 20 * 60,
        },
        "metric": {
            "normalize_by": "available_budget",
            "temperature": tune.grid_search(temperature),
            "n_knapsack_solvers": n_knapsack_solvers,
            "gurobi_timeout": 60 * gurobi_timeout_minutes,
            "gurobi_threads": gurobi_threads,
        },
        "logs": {
            "verbose": False,
            "save": True,
        },
        "blocks": {
            "initial_num": num_blocks,
            "max_num": num_blocks,
        },
    }

    experiment_analysis = tune.run(
        run_and_report,
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

    all_trial_paths = experiment_analysis._get_trial_paths()
    experiment_dir = Path(all_trial_paths[0]).parent

    rdf = load_ray_experiment(experiment_dir)

    return rdf


def grid_offline_heterogeneity_knob(
    num_tasks: List[int],
    num_blocks: List[int],
    data_path: str = "",
    optimal: bool = False,
    metric_recomputation_period: int = 10,
    parallel: bool = False,
    gurobi_timeout_minutes: int = 1,
):

    metrics = [
        # SIMPLEX,
        DOMINANT_SHARES,
        # FLAT_RELEVANCE,
        # OVERFLOW_RELEVANCE,
        ARGMAX_KNAPSACK,
    ]

    # frequencies = [f"frequencies-{p}.yaml" for p in P_GRID]
    frequencies = ["frequencies-0.95.yaml"]

    num_blocks = tune.grid_search(num_blocks)
    block_selection_policies = ["RandomBlocks"]
    temperature = [-1]

    n_knapsack_solvers = os.cpu_count() // 8 if parallel else 1
    gurobi_threads = os.cpu_count() // 4

    config = {}
    config["omegaconf"] = {
        "scheduler": {
            "method": "offline",
            "metric": tune.grid_search(metrics),
            "metric_recomputation_period": metric_recomputation_period,
            "log_warning_every_n_allocated_tasks": 250,
            "scheduler_timeout_seconds": 20 * 60,
        },
        "metric": {
            "normalize_by": "available_budget",
            "temperature": tune.grid_search(temperature),
            "n_knapsack_solvers": n_knapsack_solvers,
            "gurobi_timeout": 60 * gurobi_timeout_minutes,
            "gurobi_threads": gurobi_threads,
        },
        "logs": {
            "verbose": False,
            "save": True,
        },
        "blocks": {
            "initial_num": num_blocks,
            "max_num": num_blocks,
        },
        "tasks": {
            "data_path": data_path,
            "tasks_path": tune.grid_search(
                ["tasks"] + [f"tasks-mu10-sigma{s}" for s in [1, 2, 4, 6, 10]]
            ),
            "frequencies_path": tune.grid_search(frequencies),
            "block_selection_policy": tune.grid_search(block_selection_policies),
            "sampling": POISSON,
            "initial_num": tune.grid_search(num_tasks),
        },
    }

    experiment_analysis = tune.run(
        run_and_report,
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

    all_trial_paths = experiment_analysis._get_trial_paths()
    experiment_dir = Path(all_trial_paths[0]).parent

    rdf = load_ray_experiment(experiment_dir)

    def get_variance(path):
        _, d = path.split("-")
        p = float(d.replace(".yaml", ""))
        return (1 - p) / p ** 2

    def get_block_std(path):
        if "sigma" not in path:
            return 0
        return float(path.split("sigma")[1])

    rdf["variance"] = rdf["task_frequencies_path"].apply(get_variance)
    rdf["block_std"] = rdf["tasks_path"].apply(get_block_std)
    return rdf
    # return experiment_analysis.dataframe()


def grid_online(
    custom_config: str = "time_based_budget_unlocking/privatekube/base.yaml",
    scheduler_scheduling_time=[1],
    metric_recomputation_period=100,
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
    # temperature = [0.001, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000]
    temperature = [0.01]

    # Fully unlocked case
    # n = [1]
    # data_lifetime = [0.001]

    # Progressive unlocking
    n = [1_000]
    data_lifetime = [5]

    avg_number_tasks_per_block = [100]
    max_blocks = [20]
    initial_blocks = [10]
    block_selection_policies = ["LatestBlocksFirst"]

    # data_path = "mixed_curves"
    data_path = "mixed_curves_profits"

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

    config["omegaconf"] = {
        "scheduler": {
            "metric_recomputation_period": metric_recomputation_period,
            # "log_warning_every_n_allocated_tasks": 500,
            "scheduler_timeout_seconds": 20 * 60,
            DATA_LIFETIME: tune.grid_search(data_lifetime),
            SCHEDULING_WAIT_TIME: tune.grid_search(scheduler_scheduling_time),
            METHOD: "batch",
            METRIC: tune.grid_search(scheduler_metrics),
            N: tune.grid_search(n),
        },
        "metric": {
            "normalize_by": "available_budget",
            "temperature": tune.grid_search(temperature),
            "n_knapsack_solvers": 1,
        },
        "logs": {
            "verbose": False,
            "save": True,
        },
        "blocks": {
            "initial_num": tune.grid_search(initial_blocks),
            "max_num": tune.grid_search(max_blocks),
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
