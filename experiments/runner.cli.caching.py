import os
import typer
from experiments.ray_runner import grid_online
from precycle.utils.utils import REPO_ROOT
from notebooks.caching.utils import get_df, analyze_monoblock, analyze_multiblock
app = typer.Typer()


def get_paths(dataset):
    tasks_path_prefix = REPO_ROOT.joinpath(f"data/{dataset}/{dataset}_workload/")
    blocks_path_prefix = REPO_ROOT.joinpath(f"data/{dataset}/{dataset}_data/")
    blocks_metadata = str(blocks_path_prefix.joinpath("blocks1/metadata.json"))
    blocks_path = str(blocks_path_prefix.joinpath("blocks1"))
    return blocks_path, blocks_metadata, tasks_path_prefix


# Covid19 Dataset Experiments
def caching_monoblock_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["1blocks_34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]
    logs_dir = f"kelly/{dataset}/monoblock"

    grid_online(
        global_seed=64,
        logs_dir=logs_dir,
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["NoCuts"],
        mechanism=["Hybrid", "Laplace", "PMW"],
        # mechanism=["PMW"],
        initial_blocks=[1],
        max_blocks=[1],
        avg_num_tasks_per_block=[1e3],
        max_tasks=[1e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0, 1],
        heuristic=["bin_visits:100-5"],
        variance_reduction=[True],
        log_every_n_tasks=100,
        learning_rate=[0.2],
        bootstrapping=[False],
        exact_match_caching=[False, True],
    )
    analyze_monoblock(logs_dir)


def caching_static_multiblock_laplace_vs_hybrid_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["1:2:4:8:16:32:64:128blocks_34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir=f"kelly{dataset}/static-multiblock_laplace_vs_hybrid",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        mechanism=["Laplace"], #, "Hybrid"],
        initial_blocks=[150],
        max_blocks=[150],
        block_selection_policy=["RandomBlocks"],
        avg_num_tasks_per_block=[2e3],
        max_tasks=[300e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[1],
        heuristic=["bin_visits:100-5"],
        variance_reduction=[True],
        log_every_n_tasks=500,
        learning_rate=[0.2],
        bootstrapping=[False],
    )


def caching_static_multiblock_heuristics_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["1:2:4:8:16:32:64:128blocks_34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir="{dataset}/static-multiblock_heuristics",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        mechanism=["Hybrid"],
        initial_blocks=[150],
        max_blocks=[150],
        block_selection_policy=["RandomBlocks"],
        avg_num_tasks_per_block=[2e3],
        max_tasks=[300e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[1],
        heuristic=[
            "bin_visits:1-1",
            "bin_visits:100-5",
            "bin_visits:1000-100",
            "bin_visits:10000-100",
        ],
        variance_reduction=[True],
        log_every_n_tasks=500,
        learning_rate=[0.2],
        bootstrapping=[False],
    )


def caching_static_multiblock_learning_rate_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["1:2:4:8:16:32:64:128blocks_34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir="{dataset}/static-multiblock_learning_rate",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        mechanism=["Hybrid"],
        initial_blocks=[150],
        max_blocks=[150],
        block_selection_policy=["RandomBlocks"],
        avg_num_tasks_per_block=[2e3],
        max_tasks=[300e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[1],
        heuristic=["bin_visits:100-5"],
        variance_reduction=[True],
        log_every_n_tasks=500,
        learning_rate=[0.05, 0.1, 0.15, 0.2],
        bootstrapping=[False],
    )


def caching_streaming_multiblock_laplace_vs_hybrid_covid19(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["1:1:1:2:4:8:16:32:64:128blocks_34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir="{dataset}/streaming_multiblock_laplace_vs_hybrid",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        mechanism=["Hybrid"],
        # mechanism=["Laplace"],
        # mechanism=["Laplace", "Hybrid"],
        initial_blocks=[1],
        max_blocks=[150],
        block_selection_policy=["LatestBlocks"],
        avg_num_tasks_per_block=[2e3],
        max_tasks=[300e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0, 0.5, 1, 1.5],
        heuristic=["bin_visits:100-5"],
        variance_reduction=[True],
        log_every_n_tasks=500,
        learning_rate=[0.2],
        bootstrapping=[True, False],
    )


# Citibike Dataset Experiments
def caching_monoblock_citibike(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["1blocks_2485queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir=f"new/{dataset}/monoblock",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        mechanism=["Hybrid", "Laplace", "PMW"],
        # mechanism=["Hybrid"],
        initial_blocks=[170],
        max_blocks=[170],
        block_selection_policy=[
            "LatestBlocks"
        ],  # Doing this to choose the 169th block instead of the first one
        avg_num_tasks_per_block=[20e3],
        max_tasks=[20e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0],
        heuristic=["bin_visits:10-5"],
        variance_reduction=[True],
        log_every_n_tasks=100,
        learning_rate=[1],
        bootstrapping=[False],
    )


def caching_static_multiblock_laplace_vs_hybrid_citibike(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["1:2:4:8:16:32:64:128blocks_2485queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir=f"{dataset}/static-multiblock_laplace_vs_hybrid",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        # mechanism=["Laplace", "Hybrid"],
        mechanism=["Hybrid"],
        initial_blocks=[188],
        max_blocks=[188],
        block_selection_policy=["RandomBlocks"],
        avg_num_tasks_per_block=[2e3],
        max_tasks=[376e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0],
        heuristic=["bin_visits:10-5"],
        variance_reduction=[True],
        log_every_n_tasks=500,
        learning_rate=[1],
        bootstrapping=[False],
    )


def caching_static_multiblock_heuristics_citibike(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["1:2:4:8:16:32:64:128blocks_2485queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir="{dataset}/static-multiblock_heuristics",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        mechanism=["Hybrid"],
        initial_blocks=[188],
        max_blocks=[188],
        block_selection_policy=["RandomBlocks"],
        avg_num_tasks_per_block=[2e3],
        max_tasks=[300e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0],
        heuristic=[
            "bin_visits:1-1",
            "bin_visits:100-5",
            "bin_visits:1000-100",
            "bin_visits:10000-100",
        ],
        variance_reduction=[True],
        log_every_n_tasks=500,
        learning_rate=[1],
        bootstrapping=[False],
    )


def caching_static_multiblock_learning_rate_citibike(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["1:2:4:8:16:32:64:128blocks_2485queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir="{dataset}/static-multiblock_learning_rate",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        mechanism=["Hybrid"],
        initial_blocks=[188],
        max_blocks=[188],
        block_selection_policy=["RandomBlocks"],
        avg_num_tasks_per_block=[2e3],
        max_tasks=[300e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0],
        heuristic=["bin_visits:100-5"],
        variance_reduction=[True],
        log_every_n_tasks=500,
        learning_rate=[0.05, 0.1, 0.15, 0.2],
        bootstrapping=[False],
    )


def caching_streaming_multiblock_laplace_vs_hybrid_citibike(dataset):
    blocks_path, blocks_metadata, tasks_path_prefix = get_paths(dataset)
    task_paths = ["1:1:1:2:4:8:16:32:64:128blocks_2485queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir=f"pickle/{dataset}/streaming_multiblock_laplace_vs_hybrid",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        # mechanism=["Hybrid"],
        # mechanism=["Laplace"],
        mechanism=["Laplace", "Hybrid"],
        initial_blocks=[1],
        max_blocks=[188],
        block_selection_policy=["LatestBlocks"],
        avg_num_tasks_per_block=[2e3],
        max_tasks=[376e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0],
        heuristic=["bin_visits:10-5"],
        variance_reduction=[True],
        log_every_n_tasks=500,
        learning_rate=[1],
        bootstrapping=[True],
    )


@app.command()
def run(
    exp: str = "caching_monoblock", dataset: str = "covid19", loguru_level: str = "INFO"
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}_{dataset}"](dataset)


if __name__ == "__main__":
    app()
