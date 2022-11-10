import os
from pathlib import Path
from typing import Any, Dict, List
import ray
import mlflow
from loguru import logger
from ray import tune
from experiments.ray.analysis import load_ray_experiment
from privacypacking.config import Config
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import RAY_LOGS, LOGS_PATH
import datetime


def run_and_report(config: dict, replace=False) -> None:
    if config["omegaconf"]["logs"]["mlflow"]:
        omegaconf = config["omegaconf"]
        run_name = f"{omegaconf['repetition']}/{omegaconf['scheduler']['cache']}/{omegaconf['scheduler']['planner']}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("config", config)
            metrics = Simulator(Config(config)).run()
    else:
        metrics = Simulator(Config(config)).run()

    tune.report(**metrics)


def grid_online(
    scheduler_scheduling_time: List[float],
    metric_recomputation_period: List[int],
    scheduler_metrics: List[int],
    n: List[float],
    initial_blocks: List[int],
    initial_tasks: List[int],
    max_blocks: List[int],
    tasks_path: List[str],
    queries_path: List[str],
    blocks_path: str,
    blocks_metadata: str,
    tasks_sampling: str,
    data_lifetime: List[float],
    task_lifetime: List[int],
    planner: List[str],  # Options = {DynamicProgrammingPlanner, PerBlockPlanner}
    cache: List[str],  # Options = {DeterministicCache, ProbabilisticCache}
    enable_caching: List[bool],
    enable_dp: List[bool],
    avg_num_tasks_per_block: List[int] = [100],
    repetitions: int = 1,
    enable_random_seed: bool = False,
):
    # Progressive unlocking
    # n = [1_000]
    exp_name = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    print(exp_name)

    enable_mlflow = True
    block_selection_policy = ["LatestBlocksFirst"]
    config = {
        "omegaconf": {
            "epsilon": 10,
            "delta": 0.00001,
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
                "cache": tune.grid_search(cache),
                "scheduling_wait_time": tune.grid_search(scheduler_scheduling_time),
                "method": "batch",
                "metric": tune.grid_search(scheduler_metrics),
                "n": tune.grid_search(n),
                "enable_caching": tune.grid_search(enable_caching),
                "enable_dp": tune.grid_search(enable_dp),
            },
            "metric": {
                "normalize_by": "available_budget",
                "n_knapsack_solvers": 1,
            },
            "logs": {
                "verbose": False,
                "save": True,
                "mlflow": enable_mlflow,
            },
            "blocks": {
                "initial_num": tune.grid_search(initial_blocks),
                "max_num": tune.grid_search(max_blocks),
                "path": blocks_path,
                "metadata": blocks_metadata,
            },
            "tasks": {
                "sampling": tasks_sampling,
                "initial_num": tune.grid_search(initial_tasks),
                "path": tune.grid_search(tasks_path),
                "queries_path": tune.grid_search(queries_path),
                "block_selection_policy": tune.grid_search(block_selection_policy),
                "avg_num_tasks_per_block": tune.grid_search(avg_num_tasks_per_block),
            },
            "repetition": tune.grid_search(list(range(1, repetitions + 1))),
        }
    }

    if enable_mlflow:
        path = LOGS_PATH.joinpath("mlruns")
        os.environ["MLFLOW_TRACKING_URI"] = str(path)
        os.environ["MLFLOW_EXPERIMENT_NAME"] = exp_name
        mlflow.create_experiment(exp_name, artifact_location=str(path))

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
            # tune.integration.mlflow.MLflowLoggerCallback(
            #     experiment_name=f"{datetime.now().strftime('%m%d-%H%M%S')}",
            # ),
        ],
        progress_reporter=ray.tune.CLIReporter(
            metric_columns=[
                "n_allocated_tasks",
                "total_tasks",
                "realized_profit",
                "budget_utilization",
                "realized_budget",
            ],
            parameter_columns={
                "omegaconf/scheduler/scheduling_wait_time": "T",
                "omegaconf/scheduler/enable_caching": "enable_caching",
                "omegaconf/scheduler/planner": "planner",
                "omegaconf/scheduler/cache": "cache",
                "omegaconf/scheduler/data_lifetime": "lifetime",
                "omegaconf/scheduler/metric": "metric",
            },
            max_report_frequency=60,
        ),
    )
    # all_trial_paths = experiment_analysis._get_trial_paths()
    # experiment_dir = Path(all_trial_paths[0]).parent


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
