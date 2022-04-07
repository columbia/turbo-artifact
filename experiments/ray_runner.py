import os
from datetime import datetime
from pathlib import Path
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
from privacypacking.utils.generate_curves import P_GRID
from privacypacking.utils.utils import RAY_LOGS


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

    config = {}
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


def grid_offline_heterogeneity_knob(
    num_tasks: List[int],
    num_blocks: List[int],
    data_path: str = "",
    optimal: bool = False,
    metric_recomputation_period: int = 10,
    parallel: bool = False,
    gurobi_timeout_minutes: int = 1,
    block_axis=False,
    alpha_axis=True,
):

    metrics = [
        SIMPLEX,
        DOMINANT_SHARES,
        # FLAT_RELEVANCE,
        # OVERFLOW_RELEVANCE,
        ARGMAX_KNAPSACK,
    ]

    # tasks_paths = ["tasks"]

    tasks_paths = (
        [f"tasks-mu10-sigma{s}" for s in [0, 1, 2, 4, 6, 10]]
        if block_axis
        else [f"tasks-mu10-sigma0"]
    )
    frequencies = (
        [f"frequencies-{p}.yaml" for p in P_GRID]
        if alpha_axis
        else ["frequencies-0.95.yaml"]
    )

    num_blocks = tune.grid_search(num_blocks)
    block_selection_policies = ["RandomBlocks"]
    temperature = [-1]

    n_knapsack_solvers = os.cpu_count() // 8 if parallel else 1
    gurobi_threads = os.cpu_count() // 4

    config = {}
    config["omegaconf"] = {
        "global_seed": 0,
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
            "tasks_path": tune.grid_search(tasks_paths),
            "frequencies_path": tune.grid_search(frequencies),
            "block_selection_policy": tune.grid_search(block_selection_policies),
            "sampling": "poisson",
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

    def get_alpha_std(path):
        _, d = path.split("-")
        p = float(d.replace(".yaml", ""))
        return p

    def get_block_std(path):
        if "sigma" not in path:
            return 0
        return float(path.split("sigma")[1])

    # rdf["variance"] = rdf["task_frequencies_path"].apply(get_variance)
    rdf["alpha_std"] = rdf["task_frequencies_path"].apply(get_alpha_std)
    rdf["block_std"] = rdf["tasks_path"].apply(get_block_std)
    return rdf
    # return experiment_analysis.dataframe()


def grid_online(
    scheduler_scheduling_time: List[int],
    initial_blocks: List[int],
    max_blocks: List[int],
    metric_recomputation_period: List[int],
    data_path: List[str],
    tasks_sampling: str,
    data_lifetime: List[int],
    avg_num_tasks_per_block: List[int] = [100],
):
    # ray.init(log_to_driver=False)
    scheduler_metrics = [
        # SOFT_KNAPSACK,
        ARGMAX_KNAPSACK,
        # BATCH_OVERFLOW_RELEVANCE,
        #  FLAT_RELEVANCE,
        # DYNAMIC_FLAT_RELEVANCE,
        FCFS,
        # # VECTORIZED_BATCH_OVERFLOW_RELEVANCE,
        DOMINANT_SHARES,
    ]

    temperature = [0.01]

    # Fully unlocked case
    # n = [1]
    # data_lifetime = [0.001]

    # Progressive unlocking
    n = [1_000]
    # data_lifetime = [5]

    block_selection_policy = ["LatestBlocksFirst"]
    config = {}

    config["omegaconf"] = {
        "scheduler": {
            "metric_recomputation_period": tune.grid_search(
                metric_recomputation_period
            ),
            # "log_warning_every_n_allocated_tasks": 50,
            "scheduler_timeout_seconds": 20 * 60 * 60,
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
            "block_selection_policy": tune.grid_search(block_selection_policy),
            "avg_num_tasks_per_block": tune.grid_search(avg_num_tasks_per_block),
        },
    }
    logger.info(f"Tune config: {config}")

    experiment_analysis = tune.run(
        run_and_report,
        config=config,
        resources_per_trial={"cpu": 1},
        # resources_per_trial={"cpu": 32},
        local_dir=RAY_LOGS,
        resume=False,
        verbose=1,
        callbacks=[
            CustomLoggerCallback(),
            tune.logger.JsonLoggerCallback(),
            tune.integration.mlflow.MLflowLoggerCallback(
                experiment_name=f"grid-online-{datetime.now().strftime('%m%d-%H%M%S')}",
            ),
        ],
        progress_reporter=ray.tune.CLIReporter(
            metric_columns=["n_allocated_tasks", "total_tasks", "realized_profit"],
            parameter_columns={
                "omegaconf/scheduler/scheduling_wait_time": "T",
                "omegaconf/scheduler/data_lifetime": "lifetime",
                "omegaconf/scheduler/metric": "metric",
                # "omegaconf/metric/temperature": "temperature",
            },
            max_report_frequency=60,
        ),
    )
    all_trial_paths = experiment_analysis._get_trial_paths()
    experiment_dir = Path(all_trial_paths[0]).parent
    rdf = load_ray_experiment(experiment_dir)
    return rdf


class CustomLoggerCallback(tune.logger.LoggerCallback):
    """Custom logger interface"""

    def __init__(self, metrics=["scheduler_metric"]) -> None:
        self.metrics = ["n_allocated_tasks", "realized_profit"]
        self.metrics.extend(metrics)
        super().__init__()

    def log_trial_result(self, iteration: int, trial: Any, result: Dict):
        logger.info([f"{key}: {result[key]}" for key in self.metrics])
        return

    def on_trial_complete(self, iteration: int, trials: List, trial: Any, **info):
        return
