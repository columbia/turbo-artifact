import os
import ray
import mlflow
import datetime
from ray import tune
from loguru import logger
from typing import List, Any, Dict
from precycle.utils.utils import LOGS_PATH, RAY_LOGS
from precycle.run_simulation import Simulator


def run_and_report(config: dict, replace=False) -> None:
    logs = Simulator(config).run()
    tune.report(**logs)


def grid_online(
    initial_blocks: List[int],
    initial_tasks: List[int],
    max_blocks: List[int],
    logs_dir: str,
    tasks_path: List[str],
    blocks_path: str,
    blocks_metadata: str,
    planner: List[str],
    cache: List[str],
    avg_num_tasks_per_block: List[int] = [100],
    max_tasks: List[int] = [None],
    enable_random_seed: bool = False,
    global_seed: int = 64,
    alpha: List[int] = [0.05],
    beta: List[int] = [0.0001],
    heuristic: str = "total_updates_counts",
    heuristic_value: List[float] = [100],
    zipf_k: List[int] = [0.5],
    max_pmw_k: List[int] = 10,
):
    exp_name = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    enable_mlflow = True
    config = {
        "mock": True,
        "global_seed": global_seed,
        "enable_random_seed": enable_random_seed,
        "cache": {
            "type": tune.grid_search(cache),
            "probabilistic_cfg": {
                "alpha": tune.grid_search(alpha),
                "beta": tune.grid_search(beta),
                "max_pmw_k": tune.grid_search(max_pmw_k),
            },
            "pmw_cfg": {
                "heuristic": heuristic,
                "heuristic_value": tune.grid_search(heuristic_value),
            },
        },
        "planner": {
            "method": tune.grid_search(planner),
        },
        "budget_accountant": {"epsilon": 3, "delta": 1e-07},
        "blocks": {
            "initial_num": tune.grid_search(initial_blocks),
            "max_num": tune.grid_search(max_blocks),
            "arrival_interval": 1,
            "block_data_path": blocks_path,
            "block_metadata_path": blocks_metadata,
        },
        "tasks": {
            "path": tune.grid_search(tasks_path),
            "avg_num_tasks_per_block": tune.grid_search(avg_num_tasks_per_block),
            "max_num": tune.grid_search(max_tasks),
            "initial_num": tune.grid_search(initial_tasks),
            "zipf_k": tune.grid_search(zipf_k),
        },
        "logs": {
            "verbose": False,
            "save": True,
            "mlflow": True,
            "loguru_level": "INFO",
        },
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
        # resources_per_trial={"cpu": 1},
        resources_per_trial={"cpu": 1},
        local_dir=RAY_LOGS.joinpath(logs_dir),
        resume=False,
        verbose=1,
        callbacks=[
            CustomLoggerCallback(),
            tune.logger.JsonLoggerCallback(),
        ],
        progress_reporter=ray.tune.CLIReporter(
            metric_columns=[
                "n_allocated_tasks",
                "total_tasks",
            ],
            parameter_columns={
                "planner/method": "planner",
                "cache/type": "cache",
                "tasks/zipf_k": "zipf_k",
            },
            max_report_frequency=60,
        ),
    )
    # all_trial_paths = experiment_analysis._get_trial_paths()
    # experiment_dir = Path(all_trial_paths[0]).parent


class CustomLoggerCallback(tune.logger.LoggerCallback):

    """Custom logger interface"""

    def __init__(self, metrics=[]) -> None:
        self.metrics = ["n_allocated_tasks"]
        self.metrics.extend(metrics)
        super().__init__()

    def log_trial_result(self, iteration: int, trial: Any, result: Dict):
        logger.info([f"{key}: {result[key]}" for key in self.metrics])
        return

    def on_trial_complete(self, iteration: int, trials: List, trial: Any, **info):
        return
