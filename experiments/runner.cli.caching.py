import os
import typer
import multiprocessing
from experiments.ray_runner import grid_online
from precycle.utils.utils import REPO_ROOT
from notebooks.caching.utils import get_df, analyze_monoblock, analyze_multiblock
from copy import deepcopy

app = typer.Typer()


def get_paths(dataset):
    tasks_path_prefix = REPO_ROOT.joinpath(f"data/{dataset}/{dataset}_workload/")
    blocks_path_prefix = REPO_ROOT.joinpath(f"data/{dataset}/{dataset}_data/")
    blocks_metadata = str(blocks_path_prefix.joinpath("blocks/metadata.json"))
    blocks_path = str(blocks_path_prefix.joinpath("blocks"))
    return blocks_path, blocks_metadata, tasks_path_prefix


def experiments_start_and_join(experiments):
    for p in experiments:
        p.start()
    for p in experiments:
        p.join()


# Covid19 Dataset Experiments
def caching_monoblock_PMW_vs_Laplace_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    block_requests_pattern = [1]

    logs_dir = f"{dataset}/monoblock_laplace_vs_pmw"
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "planner": ["NoCuts"],
        "mechanism": ["Laplace", "PMW"],
        "initial_blocks": [1],
        "max_blocks": [1],
        "avg_num_tasks_per_block": [2e4],
        "max_tasks": [2e4],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [1],
        "heuristic": [""],
        "variance_reduction": [True],
        "log_every_n_tasks": 100,
        "learning_rate": [0.2],
        "bootstrapping": [False],
        "exact_match_caching": [False],
    }

    grid_online(**config)
    analyze_monoblock(logs_dir)


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
        "mechanism": ["Hybrid", "PMW"],
        "initial_blocks": [1],
        "max_blocks": [1],
        "avg_num_tasks_per_block": [7e4],
        "max_tasks": [7e4],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [0, 1],
        "heuristic": ["bin_visits:100-5"],
        "variance_reduction": [True],
        "log_every_n_tasks": 100,
        "learning_rate": [0.2],
        "bootstrapping": [False],
        "exact_match_caching": [True],
    }
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    config["mechanism"] = ["Laplace"]
    config["heuristic"] = [""]
    config["exact_match_caching"] = [True, False]
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)
    analyze_monoblock(logs_dir)


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
            "bin_visits:100-5",
            "bin_visits:1000-100",
            "bin_visits:10000-100",
        ],
        "variance_reduction": [True],
        "log_every_n_tasks": 100,
        "learning_rate": [0.2],
        "bootstrapping": [False],
        "exact_match_caching": [True],
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
    analyze_monoblock(logs_dir)


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
        "variance_reduction": [True],
        "log_every_n_tasks": 100,
        "learning_rate": [0.05, 0.1, 0.2, 0.4, 1],
        "bootstrapping": [False],
        "exact_match_caching": [True],
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
    analyze_monoblock(logs_dir)


def caching_static_multiblock_laplace_vs_hybrid_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    # task_paths = ["1:2:4:8:16:32:64:128blocks_34425queries.privacy_tasks.csv"]
    # task_paths = ["1:2:3:4:5:6:7:8:9:10blocks_34425queries.privacy_tasks.csv"]
    task_paths = ["34425queries.privacy_tasks.csv"]
    block_requests_pattern = list(range(1, 51))
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    logs_dir = f"help/{dataset}/static_multiblock/laplace_vs_hybrid"
    experiments = []
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "planner": ["MinCuts"],
        "mechanism": ["Laplace"],
        "initial_blocks": [50],
        "max_blocks": [50],
        "avg_num_tasks_per_block": [2e3],
        "max_tasks": [100e3],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [0, 1],
        "heuristic": [""],
        "variance_reduction": [True],
        "log_every_n_tasks": 500,
        "learning_rate": [0.2],
        "bootstrapping": [False],
        "exact_match_caching": [True],
    }
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    config["planner"] = ["NoCuts"]
    config["mechanism"] = ["Laplace"]
    config["heuristic"] = [""]
    config["exact_match_caching"] = [True, False]
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    config["exact_match_caching"] = [True]
    config["planner"] = ["MinCuts"]
    config["mechanism"] = ["Hybrid"]
    config["heuristic"] = ["bin_visits:200-50"]
    config["learning_rate"] = [0.2, 0.1]
    config["bootstrapping"] = [False]

    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)
    analyze_multiblock(logs_dir)


# def caching_static_multiblock_heuristics_covid19(dataset):
#     blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
#     # task_paths = ["1:2:4:8:16:32:64:128blocks_34425queries.privacy_tasks.csv"]
#     task_paths = ["34425queries.privacy_tasks.csv"]
#     task_paths = [
#         str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
#     ]
#     block_requests_pattern = list(range(1,101))

#     logs_dir = "{dataset}/static-multiblock_heuristics"
#     experiments = []
#     config = {
#         "global_seed": 64,
#         "logs_dir": logs_dir,
#         "tasks_path": task_paths,
#         "blocks_path": blocks_path,
#         "blocks_metadata": blocks_metadata,
#         "block_requests_pattern": block_requests_pattern,
#         "planner": ["MinCuts"],
#         "mechanism": ["Hybrid"],
#         "initial_blocks": [150],
#         "max_blocks": [150],
#         "avg_num_tasks_per_block": [2e3],
#         "max_tasks": [300e3],
#         "initial_tasks": [0],
#         "alpha": [0.05],
#         "beta": [0.001],
#         "zipf_k": [1],
#         "heuristic": ["bin_visits:1-1",
#             "bin_visits:100-5",
#             "bin_visits:1000-100",
#             "bin_visits:10000-100",
#         ],
#         "variance_reduction": [True],
#         "log_every_n_tasks": 500,
#         "learning_rate": [0.2],
#         "bootstrapping": [False],
#         "exact_match_caching": [True],
#     }
#     experiments.append(
#         multiprocessing.Process(
#             target=lambda config: grid_online(**config), args=(deepcopy(config),)
#         )
#     )
#     experiments_start_and_join(experiments)
#     analyze_multiblock(logs_dir)


def caching_static_multiblock_learning_rate_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    # task_paths = ["1:2:4:8:16:32:64:128blocks_34425queries.privacy_tasks.csv"]
    task_paths = ["34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    block_requests_pattern = list(range(1, 101))
    logs_dir = "{dataset}/static-multiblock_learning_rate"
    experiments = []
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "planner": ["MinCuts"],
        "mechanism": ["Hybrid"],
        "initial_blocks": [150],
        "max_blocks": [150],
        "avg_num_tasks_per_block": [2e3],
        "max_tasks": [300e3],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [1],
        "heuristic": [
            "bin_visits:1-1",
            "bin_visits:100-5",
            "bin_visits:1000-100",
            "bin_visits:10000-100",
        ],
        "variance_reduction": [True],
        "log_every_n_tasks": 500,
        "learning_rate": [0.05, 0.1, 0.2, 0.4],
        "bootstrapping": [False],
        "exact_match_caching": [True],
    }
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)
    analyze_multiblock(logs_dir)


def caching_streaming_multiblock_laplace_vs_hybrid_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    # task_paths = ["1:1:1:2:4:8:16:32:64:128blocks_34425queries.privacy_tasks.csv"]
    task_paths = ["34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    block_requests_pattern = [1, 1, 1, 2, 4, 8, 16, 32, 64, 128]
    logs_dir = f"{dataset}/streaming_multiblock_laplace_vs_hybrid"
    experiments = []
    config = {
        "global_seed": 64,
        "logs_dir": logs_dir,
        "tasks_path": task_paths,
        "blocks_path": blocks_path,
        "blocks_metadata": blocks_metadata,
        "block_requests_pattern": block_requests_pattern,
        "planner": ["MinCuts"],
        "mechanism": ["Laplace"],
        "initial_blocks": [1],
        "max_blocks": [150],
        "block_selection_policy": ["LatestBlocks"],
        "avg_num_tasks_per_block": [2e3],
        "max_tasks": [300e3],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [0, 1],
        "heuristic": [""],
        "variance_reduction": [True],
        "log_every_n_tasks": 500,
        "learning_rate": [0.2],
        "bootstrapping": [True],
        "exact_match_caching": [True],
    }
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    config["planner"] = ["NoCuts"]
    config["mechanism"] = ["Laplace"]
    config["heuristic"] = [""]
    config["exact_match_caching"] = [True, False]
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    config["exact_match_caching"] = [True]
    config["planner"] = ["MinCuts"]
    config["mechanism"] = ["Hybrid"]
    config["heuristic"] = ["bin_visits:100-10"]
    config["learning_rate"] = [0.2]
    config["bootstrapping"] = [False]

    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)
    analyze_multiblock(logs_dir)


# Citibike Dataset Experiments
def caching_monoblock_citibike(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["2485queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    block_requests_pattern = [1]
    logs_dir = f"{dataset}/monoblock7"
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
        "initial_blocks": [170],
        "max_blocks": [170],
        "block_selection_policy": [
            "LatestBlocks"
        ],  # Doing this to choose the 106th block instead of the first one
        "avg_num_tasks_per_block": [7e4],
        "max_tasks": [7e4],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [0],
        "heuristic": [""],
        "variance_reduction": [True],
        "log_every_n_tasks": 100,
        "learning_rate": [1],
        "bootstrapping": [False],
        "exact_match_caching": [True],
    }
    # experiments.append(
    #     multiprocessing.Process(
    #         target=lambda config: grid_online(**config), args=(deepcopy(config),)
    #     )
    # )
    config["mechanism"] = ["Hybrid"]
    config["heuristic"] = ["bin_visits:2-5"]
    config["learning_rate"] = ["0:2_5:0.5_10:0.1"]

    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    # config["mechanism"] = ["Laplace"]
    # config["heuristic"] = [""]
    # config["learning_rate"] = [None]
    # config["exact_match_caching"] = [True, False]
    # experiments.append(
    #     multiprocessing.Process(
    #         target=lambda config: grid_online(**config), args=(deepcopy(config),)
    #     )
    # )
    experiments_start_and_join(experiments)
    analyze_monoblock(logs_dir)


# def caching_static_multiblock_laplace_vs_hybrid_citibike(dataset):
#     blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
#     task_paths = ["1:2:4:8:16:32:64:128blocks_2485queries.privacy_tasks.csv"]
#     task_paths = [
#         str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
#     ]

#     grid_online(
#         global_seed=64,
#         logs_dir=f"{dataset}/static-multiblock_laplace_vs_hybrid",
#         tasks_path=task_paths,
#         blocks_path=blocks_path,
#         blocks_metadata=blocks_metadata,
#         # "block_requests_pattern": block_requests_pattern,
#         planner=["MinCuts"],
#         # mechanism=["Laplace", "Hybrid"],
#         mechanism=["Hybrid"],
#         initial_blocks=[188],
#         max_blocks=[188],
#         block_selection_policy=["RandomBlocks"],
#         avg_num_tasks_per_block=[2e3],
#         max_tasks=[376e3],
#         initial_tasks=[0],
#         alpha=[0.05],
#         beta=[0.001],
#         zipf_k=[0],
#         heuristic=["bin_visits:10-5"],
#         variance_reduction=[True],
#         log_every_n_tasks=500,
#         learning_rate=[1],
#         bootstrapping=[False],
#     )


# def caching_static_multiblock_heuristics_citibike(dataset):
#     blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
#     task_paths = ["1:2:4:8:16:32:64:128blocks_2485queries.privacy_tasks.csv"]
#     task_paths = [
#         str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
#     ]

#     grid_online(
#         global_seed=64,
#         logs_dir="{dataset}/static-multiblock_heuristics",
#         tasks_path=task_paths,
#         blocks_path=blocks_path,
#         blocks_metadata=blocks_metadata,
#         planner=["MinCuts"],
#         mechanism=["Hybrid"],
#         initial_blocks=[188],
#         max_blocks=[188],
#         block_selection_policy=["RandomBlocks"],
#         avg_num_tasks_per_block=[2e3],
#         max_tasks=[300e3],
#         initial_tasks=[0],
#         alpha=[0.05],
#         beta=[0.001],
#         zipf_k=[0],
#         heuristic=[
#             "bin_visits:1-1",
#             "bin_visits:100-5",
#             "bin_visits:1000-100",
#             "bin_visits:10000-100",
#         ],
#         variance_reduction=[True],
#         log_every_n_tasks=500,
#         learning_rate=[1],
#         bootstrapping=[False],
#     )


# def caching_static_multiblock_learning_rate_citibike(dataset):
#     blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
#     task_paths = ["1:2:4:8:16:32:64:128blocks_2485queries.privacy_tasks.csv"]
#     task_paths = [
#         str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
#     ]

#     grid_online(
#         global_seed=64,
#         logs_dir="{dataset}/static-multiblock_learning_rate",
#         tasks_path=task_paths,
#         blocks_path=blocks_path,
#         blocks_metadata=blocks_metadata,
#         planner=["MinCuts"],
#         mechanism=["Hybrid"],
#         initial_blocks=[188],
#         max_blocks=[188],
#         block_selection_policy=["RandomBlocks"],
#         avg_num_tasks_per_block=[2e3],
#         max_tasks=[300e3],
#         initial_tasks=[0],
#         alpha=[0.05],
#         beta=[0.001],
#         zipf_k=[0],
#         heuristic=["bin_visits:100-5"],
#         variance_reduction=[True],
#         log_every_n_tasks=500,
#         learning_rate=[0.05, 0.1, 0.15, 0.2],
#         bootstrapping=[False],
#     )


def caching_streaming_multiblock_laplace_vs_hybrid_citibike(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["2485queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    logs_dir = f"234/{dataset}/streaming_multiblock_laplace_vs_hybrid"
    block_requests_pattern = [1, 1, 1, 2, 4, 8, 16, 32, 64, 128]
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
        "max_blocks": [188],
        "block_selection_policy": ["LatestBlocks"],
        "avg_num_tasks_per_block": [2000],
        "max_tasks": [376000],
        "initial_tasks": [0],
        "alpha": [0.05],
        "beta": [0.001],
        "zipf_k": [0],
        "heuristic": [""],
        "variance_reduction": [True],
        "log_every_n_tasks": 500,
        "learning_rate": [1],
        "bootstrapping": [True],
        "exact_match_caching": [True],
    }
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    # config["planner"] = ["MinCuts"]
    # config["mechanism"] = ["Hybrid"]
    # config["heuristic"] = ["bin_visits:4-2", "bin_visits:4-1", "bin_visits:4-1"]
    # config["learning_rate"] = [1]
    experiments.append(
        multiprocessing.Process(
            target=lambda config: grid_online(**config), args=(deepcopy(config),)
        )
    )
    experiments_start_and_join(experiments)


@app.command()
def run(
    exp: str = "caching_monoblock", dataset: str = "covid19", loguru_level: str = "INFO"
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}_{dataset}"](dataset)


if __name__ == "__main__":
    app()
