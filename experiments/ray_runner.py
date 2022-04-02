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
    BATCH_OVERFLOW_RELEVANCE,
    DOMINANT_SHARES,
    DYNAMIC_FLAT_RELEVANCE,
    FCFS,
    FLAT_RELEVANCE,
    OVERFLOW_RELEVANCE,
    SIMPLEX,
    SOFT_KNAPSACK,
    VECTORIZED_BATCH_OVERFLOW_RELEVANCE,
)
from privacypacking.simulator.simulator import Simulator
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
        data_path: List[str],
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

    block_selection_policy = ["RandomBlocks"]
    temperature = [-1]
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
        "tasks": {
            "initial_num": tune.grid_search(num_tasks),
            "data_path": tune.grid_search(data_path),
            "block_selection_policy": tune.grid_search(block_selection_policy),
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


def grid_online(
    scheduler_scheduling_time: List[int],
    initial_blocks: List[int],
    max_blocks: List[int],
    metric_recomputation_period: List[int],
    data_path: List[str],
    tasks_sampling: bool,
    tasks_arrival_mode: str,
    avg_num_tasks_per_block: List[int],
    data_lifetime: List[int],
):
    # ray.init(log_to_driver=False)
    with open(DEFAULT_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

        scheduler_metrics = [
            # SOFT_KNAPSACK,
            ARGMAX_KNAPSACK,
            BATCH_OVERFLOW_RELEVANCE,
            #  FLAT_RELEVANCE,
            # DYNAMIC_FLAT_RELEVANCE,
            #  FCFS,
            # # VECTORIZED_BATCH_OVERFLOW_RELEVANCE,
            DOMINANT_SHARES,
        ]

        temperature = [-1]

        # Fully unlocked case
        # n = [1]
        # data_lifetime = [0.001]

        # Progressive unlocking
        n = [1_000]
        # data_lifetime = [5]

        block_selection_policy = ["LatestBlocksFirst"]
        config["omegaconf"] = {
            "scheduler": {
                "metric_recomputation_period": metric_recomputation_period,
                "log_warning_every_n_allocated_tasks": 500,
                "scheduler_timeout_seconds": 20 * 60,
                "data_lifetime": tune.grid_search(data_lifetime),
                "scheduling_wait_time": tune.grid_search(scheduler_scheduling_time),
                "method": "batch",
                "metric": tune.grid_search(scheduler_metrics),
                "n": tune.grid_search(n),
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
            "tasks": {
                "sampling": tasks_sampling,
                "data_path": tune.grid_search(data_path),
                "block_selection_policy": block_selection_policy,
                "arrival": tasks_arrival_mode,
                "avg_num_tasks_per_block": tune.grid_search(avg_num_tasks_per_block),
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
