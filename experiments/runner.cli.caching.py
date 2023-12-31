import multiprocessing
import os
import uuid
from copy import deepcopy

import time
import numpy as np
import typer
from utils import get_paths

from experiments.ray_runner import grid_online
from notebooks.caching.utils import (
    analyze_monoblock,
    analyze_multiblock,
    analyze_convergence,
    analyze_heuristics_2,
)

app = typer.Typer()


def experiments_start_and_join(experiments):
    for p in experiments:
        time.sleep(5)
        p.start()
    for p in experiments:
        p.join()


# Covid19 Dataset Experiments
def caching_monoblock_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    block_requests_pattern = [1]

    logs_dir = f"{dataset}/monoblock/laplace_vs_hybrid"
    experiments = []
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "planner": ["NoCuts"],
        "mechanism": ["Hybrid", "Laplace"],
        "initial_blocks": [1],
        "max_blocks": [1],
        "avg_num_tasks_per_block": [7e4],
        "max_tasks": [7e4],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [0, 1],
        "heuristic": ["bin_visits:100-5"],
        "log_every_n_tasks": 100,
        "learning_rate": ["0:2_50:0.5_100:0.2"],
        "bootstrapping": [False],
        "exact_match_caching": [True],  # , False],
        "tau": [0.05],
        "mlflow_experiment_id": "monoblock_covid",
        "external_update_on_cached_results": [False],
    }
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    config["mechanism"] = ["PMW"]
    config["heuristic"] = [""]
    config["exact_match_caching"] = [False]
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)
    analyze_monoblock(f"ray/{logs_dir}")


def caching_monoblock_heuristics_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    block_requests_pattern = [1]

    logs_dir = f"{dataset}/monoblock/heuristics"
    experiments = []
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "planner": ["NoCuts"],
        "mechanism": ["Hybrid"],
        "initial_blocks": [1],
        "max_blocks": [1],
        "avg_num_tasks_per_block": [7e4],
        "max_tasks": [7e4],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [1],
        "heuristic": [
            "bin_visits:1-1",
            "bin_visits:10-1",
            "bin_visits:100-1",
            "bin_visits:1000-1",
        ],
        "log_every_n_tasks": 100,
        "learning_rate": ["0:2_50:0.5_100:0.2"],
        "bootstrapping": [False],
        "exact_match_caching": [False],
        "tau": [0.05],
        "mlflow_experiment_id": "monoblock_covid_heuristics",
        "external_update_on_cached_results": [False],
    }
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    config["mechanism"] = ["Laplace"]
    config["heuristic"] = [""]
    config["exact_match_caching"] = [True]
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)
    analyze_monoblock(f"ray/{logs_dir}")


def caching_monoblock_heuristics_2_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    block_requests_pattern = [1]

    logs_dir = f"{dataset}/monoblock/heuristics2"
    experiments = []
    cl1 = [int(lr) for lr in np.linspace(0, 600, num=61)]
    cl = [int(lr) for lr in np.linspace(0, 3000, num=301)]
    bin_visits = [f"bin_visits:{c}-0" for c in cl1]
    bin_visits_adaptive = [f"bin_visits:{c}-1" for c in cl1]
    bin_visits_adaptive5 = [f"bin_visits:{c}-5" for c in cl1]
    global_visits = [f"global_visits:{c}-0" for c in cl]
    global_visits_adaptive = [f"global_visits:{c}-1" for c in cl]
    global_visits_adaptive5 = [f"global_visits:{c}-5" for c in cl]

    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "planner": ["NoCuts"],
        "mechanism": ["Hybrid"],
        "initial_blocks": [1],
        "max_blocks": [1],
        "avg_num_tasks_per_block": [7e4],
        "max_tasks": [7e4],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [1, 1.5],
        "heuristic": bin_visits
        + bin_visits_adaptive
        + global_visits
        + global_visits_adaptive
        + bin_visits_adaptive5
        + global_visits_adaptive5,
        "log_every_n_tasks": 100,
        "learning_rate": [0.2],
        "bootstrapping": [False],
        "exact_match_caching": [False],
        "tau": [0.05],
        "mlflow_experiment_id": "monoblock_covid_heuristics2",
        "external_update_on_cached_results": [False],
    }
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)
    analyze_heuristics_2(f"ray/{logs_dir}")


def caching_monoblock_learning_rates_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    block_requests_pattern = [1]

    logs_dir = f"{dataset}/monoblock/learning_rates"
    experiments = []
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "planner": ["NoCuts"],
        "mechanism": ["Hybrid"],
        "initial_blocks": [1],
        "max_blocks": [1],
        "avg_num_tasks_per_block": [7e4],
        "max_tasks": [7e4],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [1],
        "heuristic": ["bin_visits:100-5"],
        "log_every_n_tasks": 100,
        "learning_rate": [0.05, 0.1, 0.2, 0.4, 1],
        "bootstrapping": [False],
        "exact_match_caching": [False],
        "tau": [0.05],
        "mlflow_experiment_id": "monoblock_covid_learning_rates",
        "external_update_on_cached_results": [False],
    }
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    config["mechanism"] = ["Laplace"]
    config["heuristic"] = [""]
    config["learning_rate"] = [None]
    config["exact_match_caching"] = [True]
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)
    analyze_monoblock(f"ray/{logs_dir}")


def convergence_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    block_requests_pattern = [1]

    logs_dir = f"{dataset}/monoblock/convergence"
    experiments = []
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "planner": ["NoCuts"],
        "mechanism": ["Hybrid"],
        "initial_blocks": [1],
        "max_blocks": [1],
        "avg_num_tasks_per_block": [7e4],
        "max_tasks": [7e4],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [1],
        "heuristic": ["bin_visits:100-5", "bin_visits:0-0"],
        "variance_reduction": [False],
        "log_every_n_tasks": 100,
        # "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.4, 1, 2, 4],
        "learning_rate": [float(lr) for lr in np.geomspace(0.01, 10, num=20)],
        "bootstrapping": [False],
        "exact_match_caching": [False],
        "mlflow_random_prefix": [True],
        "validation_interval": 500,
        "mlflow_experiment_id": "convergence",
    }
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)
    analyze_convergence(f"ray/{logs_dir}")


def tree_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["34425queries.privacy_tasks.csv"]
    # block_requests_pattern = list(range(1, 51))

    # block_requests_pattern = [
    #     [1],
    #     [2],
    #     [4],
    #     [8],
    #     [16],
    #     [32]
    # ]

    block_requests_pattern = [
        f"dgaussian-{std}-{tmax}"
        for tmax in [1, 5, 10, 15, 20, 25, 30, 40, 50]
        for std in [1, 3, 5, 10, 15]
    ]

    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    logs_dir = f"test/{dataset}/static_multiblock/tree"
    experiments = []
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "block_selection_policy": ["LatestBlocks"],
        # "block_selection_policy": ["LatestBlocks"],
        "planner": ["MinCuts", "MaxCuts"],
        # "planner": ["MinCuts"],
        "mechanism": ["Hybrid"],
        "initial_blocks": [50],
        "max_blocks": [50],
        "avg_num_tasks_per_block": [6e3],
        "max_tasks": [300e3],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [1],
        "heuristic": ["bin_visits:100-5"],
        "variance_reduction": [True],
        "log_every_n_tasks": 500,
        "learning_rate": ["0:2_50:0.5_100:0.2"],
        "bootstrapping": [False],
        "exact_match_caching": [False],
        "mlflow_experiment_id": "tree_zipf1_new",
        "mlflow_random_prefix": [True],
        "tau": [0.05],
        "save_logs": False,
    }

    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    # config["exact_match_caching"] = [True]

    # config["planner"] = ["MinCuts"]
    # config["heuristic"] = [""]
    # experiments.append(
    #     multiprocessing.Process(
    #         target=lambda config: grid_online(**config), args=(deepcopy(config),)
    #     )
    # )
    # config["planner"] = ["MinCuts"]
    # config["mechanism"] = ["Hybrid"]
    # config["heuristic"] = ["bin_visits:100-5"]
    # config["learning_rate"] = ["0:2_50:0.5_100:0.1"]
    # config["bootstrapping"] = [False]

    # experiments.append(
    #     multiprocessing.Process(
    #         target=lambda config: grid_online(**config), args=(deepcopy(config),)
    #     )
    # )
    experiments_start_and_join(experiments)
    analyze_multiblock(logs_dir)


def caching_static_multiblock_laplace_vs_hybrid_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["34425queries.privacy_tasks.csv"]
    block_requests_pattern = list(range(1, 51))
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    logs_dir = f"{dataset}/static_multiblock/laplace_vs_hybrid"
    experiments = []
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "block_selection_policy": ["RandomBlocks"],
        "planner": ["NoCuts"],
        "mechanism": ["Laplace"],
        "initial_blocks": [50],
        "max_blocks": [50],
        "avg_num_tasks_per_block": [6e3],
        "max_tasks": [300e3],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [0, 1],
        "heuristic": [""],
        "log_every_n_tasks": 500,
        "learning_rate": [0.4],
        "bootstrapping": [False],
        "exact_match_caching": [True],
        "tau": [0.05],
        "external_update_on_cached_results": [False],
        "mlflow_experiment_id": "static_multiblock_covid",
    }

    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )

    config["exact_match_caching"] = [True]
    config["planner"] = ["MinCuts"]
    config["heuristic"] = [""]
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )

    config["planner"] = ["MinCuts"]
    config["mechanism"] = ["Hybrid"]
    config["heuristic"] = ["bin_visits:100-5"]
    config["learning_rate"] = ["0:2_50:0.5_100:0.2"]
    config["bootstrapping"] = [False]
    config["external_update_on_cached_results"] = [False]
    config["tau"] = [0.05]

    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)
    analyze_multiblock(f"ray/{logs_dir}")


def caching_streaming_multiblock_laplace_vs_hybrid_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    # block_requests_pattern = [1] * 10 + list(range(2, 51))
    block_requests_pattern = list(range(1, 51))

    logs_dir = f"{dataset}/streaming_multiblock/laplace_vs_hybrid"
    experiments = []
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "planner": ["NoCuts"],
        "mechanism": ["Laplace"],
        "initial_blocks": [1],
        "max_blocks": [50],
        "block_selection_policy": ["LatestBlocks"],
        "avg_num_tasks_per_block": [6e3],
        "max_tasks": [300e3],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [0, 1],
        "heuristic": [""],
        "log_every_n_tasks": 500,
        "learning_rate": [0.2],
        "bootstrapping": [True],
        "exact_match_caching": [True],
        "tau": [0.05],
        "external_update_on_cached_results": [False],
        "mlflow_experiment_id": "streaming_multiblock_covid",
    }
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    config["exact_match_caching"] = [True]
    config["planner"] = ["MinCuts"]
    config["heuristic"] = [""]
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )

    config["planner"] = ["MinCuts"]
    config["mechanism"] = ["Hybrid"]
    config["heuristic"] = ["bin_visits:100-5"]
    config["learning_rate"] = ["0:2_50:0.5_100:0.2"]
    config["bootstrapping"] = [True, False]
    config["external_update_on_cached_results"] = [False]
    config["tau"] = [0.05]

    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )

    experiments_start_and_join(experiments)
    analyze_multiblock(f"ray/{logs_dir}")


# Citibike Dataset Experiments
def caching_monoblock_citibike(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["2485queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    block_requests_pattern = [1]
    logs_dir = f"{dataset}/monoblock/laplace_vs_hybrid"
    experiments = []
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "planner": ["NoCuts"],
        "mechanism": ["PMW"],
        "initial_blocks": [1],
        "max_blocks": [1],
        "avg_num_tasks_per_block": [7e4],
        "max_tasks": [7e4],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [0],
        "heuristic": [""],
        "log_every_n_tasks": 100,
        "learning_rate": [1],
        "bootstrapping": [False],
        "exact_match_caching": [True],
        "tau": [0.01],
        "mlflow_experiment_id": "monoblock_citibike",
        "external_update_on_cached_results": [False],
    }
    # experiments.append(
    #     multiprocessing.Process(
    #         target=lambda config: grid_online(**config), args=(deepcopy(config),)
    #     )
    # )
    config["mechanism"] = ["Hybrid"]
    config["heuristic"] = ["bin_visits:5-1"]
    config["external_update_on_cached_results"] = [False]
    config["learning_rate"] = [4]
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    config["mechanism"] = ["Laplace"]
    config["external_update_on_cached_results"] = [False]
    config["heuristic"] = [""]
    config["learning_rate"] = [None]
    config["exact_match_caching"] = [True]  # , False]
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)
    analyze_monoblock(f"ray/{logs_dir}")


def caching_static_multiblock_laplace_vs_hybrid_citibike(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["2485queries.privacy_tasks.csv"]
    block_requests_pattern = list(range(1, 51))
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    logs_dir = f"{dataset}/static_multiblock/laplace_vs_hybrid_{str(uuid.uuid4())[:4]}"
    experiments = []
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "planner": ["NoCuts"],
        "mechanism": ["Laplace"],
        "initial_blocks": [50],
        "max_blocks": [50],
        "avg_num_tasks_per_block": [6e3],
        "max_tasks": [300e3],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [0],
        "heuristic": [""],
        "log_every_n_tasks": 500,
        "learning_rate": [2],
        "bootstrapping": [False],
        "exact_match_caching": [True],
        "tau": [0.01],
        "external_update_on_cached_results": [False],
        "mlflow_experiment_id": "multiblock_static_citibike",
    }
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    config["planner"] = ["MinCuts"]
    config["heuristic"] = [""]

    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )

    config["planner"] = ["MinCuts"]
    config["mechanism"] = ["Hybrid"]
    config["heuristic"] = ["bin_visits:5-1"]
    config["learning_rate"] = [4]
    config["bootstrapping"] = [False]
    config["external_update_on_cached_results"] = [False]
    config["tau"] = [0.01]

    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)
    analyze_multiblock(f"ray/{logs_dir}")


def caching_streaming_multiblock_laplace_vs_hybrid_citibike(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["2485queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    logs_dir = f"{dataset}/streaming_multiblock/laplace_vs_hybrid"
    # block_requests_pattern = [1] * 10 + list(range(2, 51))
    block_requests_pattern = list(range(1, 51))
    experiments = []
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "planner": ["NoCuts", "MinCuts"],
        "mechanism": ["Laplace"],
        "initial_blocks": [1],
        "max_blocks": [50],
        "block_selection_policy": ["LatestBlocks"],
        "avg_num_tasks_per_block": [6e3],
        "max_tasks": [300e3],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [0],
        "heuristic": [""],
        "log_every_n_tasks": 500,
        "learning_rate": [4],
        "bootstrapping": [True],
        "exact_match_caching": [True],
        "tau": [0.01],
        "mlflow_experiment_id": "multiblock_streaming_citibike",
        "external_update_on_cached_results": [False],
    }
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    config["planner"] = ["MinCuts"]
    config["mechanism"] = ["Hybrid"]
    config["heuristic"] = ["bin_visits:5-1"]
    config["learning_rate"] = [4]
    config["bootstrapping"] = [True, False]
    config["external_update_on_cached_results"] = [False]
    config["tau"] = [0.1]

    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)
    analyze_multiblock(f"ray/{logs_dir}")


@app.command()
def run(
    exp: str = "caching_monoblock",
    dataset: str = "covid19",
    loguru_level: str = "ERROR",
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}_{dataset}"](dataset)


if __name__ == "__main__":
    app()
