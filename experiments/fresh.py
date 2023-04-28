import os
import uuid
from typing import Dict, List

import typer
from omegaconf import OmegaConf
from ray import tune
from ray_runner import CustomLoggerCallback
from utils import get_paths

from precycle.run_simulation import Simulator
from precycle.utils.utils import DEFAULT_CONFIG_FILE, LOGS_PATH, RAY_LOGS

app = typer.Typer()


def ray_on_list(configs: List[Dict], logs_dir: str = "default"):
    def run_and_report(config: dict, replace=False) -> None:
        logs = Simulator(config).run()

        if logs:
            tune.report(**logs)

    logs_dir = configs[0]["logs"]["mlflow_experiment_id"]

    experiment_analysis = tune.run(
        run_and_report,
        config=tune.grid_search(configs),
        resources_per_trial={"cpu": 1},
        local_dir=RAY_LOGS.joinpath(logs_dir),
        resume=False,
        verbose=1,
        callbacks=[
            CustomLoggerCallback(),
            tune.logger.JsonLoggerCallback(),
        ],
        progress_reporter=tune.CLIReporter(
            metric_columns=["n_allocated_tasks", "total_tasks", "global_budget"],
            parameter_columns={
                "exact_match_caching": "exact_match_caching",
                "planner/method": "planner",
                "mechanism/type": "mechanism",
                "tasks/zipf_k": "zipf_k",
                "mechanism/probabilistic_cfg/heuristic": "heuristic",
                "mechanism/probabilistic_cfg/learning_rate": "learning_rate",
                "mechanism/probabilistic_cfg/bootstrapping": "bootstrapping",
            },
            max_report_frequency=60,
        ),
    )


@app.command()
def separate_sv(dataset: str = "covid19", loguru_level: str = "INFO"):

    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    os.environ["LOGURU_LEVEL"] = loguru_level

    base = OmegaConf.load(DEFAULT_CONFIG_FILE)

    dataset = "covid19"
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_path = "34425queries.privacy_tasks.csv"

    base.logs.mlflow_experiment_id = f"sv_debug{str(uuid.uuid4())[:0]}"

    base.blocks.block_requests_pattern = list(range(1, 51))
    base.blocks.initial_num = 50
    base.blocks.max_num = 50
    base.blocks.block_data_path = blocks_path
    base.blocks.block_metadata_path = blocks_metadata

    base.tasks.path = str(tasks_path_prefix.joinpath(task_path))
    base.tasks.block_selection_policy = "RandomBlocks"
    # base.tasks.avg_num_tasks_per_block = 6e3
    # base.tasks.max_num = 300e3

    base.tasks.avg_num_tasks_per_block = 1_000
    base.tasks.max_num = 50_000

    base.tasks.zipf_k = 0

    base.exact_match_caching = True
    base.planner.method = "MinCuts"
    base.mechanism.type = "Hybrid"
    base.mechanism.probabilistic_cfg = OmegaConf.create(
        dict(
            heuristic="bin_visits:100-5",
            learning_rate="0:2_50:0.5_100:0.2",
            bootstrapping=False,
            external_update_on_cached_results=False,
            tau=0.05,
        )
    )

    configs = []

    config = base.copy()
    config.tasks.avg_num_tasks_per_block = 6e3
    configs.append(OmegaConf.to_container(config))

    ray_on_list(configs)


if __name__ == "__main__":
    app()
