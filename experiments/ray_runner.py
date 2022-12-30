import os
import sys
import ray
import mlflow
import datetime
from ray import tune
from pathlib import Path
from loguru import logger
from typing import Any, Dict, List
from privacypacking.privacy_packing import privacypacking
from privacypacking.utils.utils import RAY_LOGS, LOGS_PATH


def run_and_report(config: dict, replace=False) -> None:
    logs = privacypacking(config)
    tune.report(**logs)


def grid_online(
    scheduler_scheduling_time: List[float],
    metric_recomputation_period: List[int],
    scheduler_metrics: List[int],
    n: List[float],
    initial_blocks: List[int],
    initial_tasks: List[int],
    max_blocks: List[int],
    logs_dir: str,
    tasks_path: List[str],
    queries_path: List[str],
    blocks_path: str,
    blocks_metadata: str,
    data_lifetime: List[float],
    task_lifetime: List[int],
    planner: List[str],  # Options = {MaxCutsPlanner}
    # optimization_objective: List[str],
    variance_reduction: List[str],
    cache: List[str],  # Options = {DeterministicCache, ProbabilisticCache}
    enable_caching: List[bool],
    enable_dp: List[bool],
    avg_num_tasks_per_block: List[int] = [100],
    max_tasks: List[int] = [4000],
    repetitions: int = 1,
    enable_random_seed: bool = False,
    alpha: List[int] = [0.005],  # For the PMW
    beta: List[int] = [0.0001],  # For the PMW
):
    # Progressive unlocking
    # n = [1_000]
    exp_name = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    enable_mlflow = False
    block_selection_policy = ["LatestBlocksFirst"]
    omegaconf = {
        "epsilon": 10,
        "delta": 1e-07,
        "enable_random_seed": enable_random_seed,
        "scheduler": {
            "metric_recomputation_period": tune.grid_search(
                metric_recomputation_period
            ),
            # "log_warning_every_n_allocated_tasks": 500,
            "scheduler_timeout_seconds": 20 * 60,
            "data_lifetime": tune.grid_search(data_lifetime),
            "task_lifetime": tune.grid_search(task_lifetime),
            "planner": tune.grid_search(planner),
            # "optimization_objective": tune.grid_search(optimization_objective),
            "variance_reduction": tune.grid_search(variance_reduction),
            "cache": tune.grid_search(cache),
            "scheduling_wait_time": tune.grid_search(scheduler_scheduling_time),
            "method": "batch",
            "metric": tune.grid_search(scheduler_metrics),
            "n": tune.grid_search(n),
            "enable_caching": tune.grid_search(enable_caching),
            "enable_dp": tune.grid_search(enable_dp),
            "pmw_cfg": {
                "alpha": tune.grid_search(alpha),
                "beta": tune.grid_search(beta),
            },
        },
        "metric": {
            "normalize_by": "available_budget",
            "n_knapsack_solvers": 1,
        },
        "logs": {
            "verbose": False,
            "save": True,
            "mlflow": enable_mlflow,
            "loguru_level": "INFO",
        },
        "blocks": {
            "initial_num": tune.grid_search(initial_blocks),
            "max_num": tune.grid_search(max_blocks),
            "path": blocks_path,
            "metadata": blocks_metadata,
        },
        "tasks": {
            "initial_num": tune.grid_search(initial_tasks),
            "path": tune.grid_search(tasks_path),
            "queries_path": tune.grid_search(queries_path),
            "block_selection_policy": tune.grid_search(block_selection_policy),
            "avg_num_tasks_per_block": tune.grid_search(avg_num_tasks_per_block),
            "max_num": tune.grid_search(max_tasks),
        },
        "repetition": tune.grid_search(list(range(1, repetitions + 1))),
    }

    if enable_mlflow:
        path = LOGS_PATH.joinpath("mlruns")
        os.environ["MLFLOW_TRACKING_URI"] = str(path)
        os.environ["MLFLOW_EXPERIMENT_NAME"] = exp_name
        mlflow.create_experiment(exp_name, artifact_location=str(path))

    logger.info(f"Tune config: {omegaconf}")

    experiment_analysis = tune.run(
        run_and_report,
        config=omegaconf,
        # resources_per_trial={"cpu": 1},
        resources_per_trial={"cpu": 32},
        local_dir=RAY_LOGS.joinpath(logs_dir),
        resume=False,
        verbose=1,
        callbacks=[
            CustomLoggerCallback(),
            tune.logger.JsonLoggerCallback(),
            # tune.integration.mlflow.MLflowLoggerCallback(
            #     experiment_name=f"{datetime.now().strftime('%m%d-%H%M%S')}",
            # ),
        ],
        progress_reporter=ray.tune.CLIReporter(
            metric_columns=[
                "n_allocated_tasks",
                "total_tasks",
                # "realized_profit",
            ],
            parameter_columns={
                "scheduler/scheduling_wait_time": "T",
                "scheduler/enable_caching": "enable_caching",
                "scheduler/planner": "planner",
                # "scheduler/optimization_objective": "optimization_objective",
                "scheduler/variance_reduction": "variance_reduction",
                "scheduler/cache": "cache",
                "scheduler/data_lifetime": "lifetime",
                "scheduler/metric": "metric",
            },
            max_report_frequency=60,
        ),
    )
    # all_trial_paths = experiment_analysis._get_trial_paths()
    # experiment_dir = Path(all_trial_paths[0]).parent


class CustomLoggerCallback(tune.logger.LoggerCallback):

    """Custom logger interface"""

    def __init__(self, metrics=["scheduler_metric"]) -> None:
        self.metrics = ["n_allocated_tasks"]  # , "realized_profit"]
        self.metrics.extend(metrics)
        super().__init__()

    def log_trial_result(self, iteration: int, trial: Any, result: Dict):
        logger.info([f"{key}: {result[key]}" for key in self.metrics])
        return

    def on_trial_complete(self, iteration: int, trials: List, trial: Any, **info):
        return
