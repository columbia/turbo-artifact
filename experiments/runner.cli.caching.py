import os
import typer
from experiments.ray_runner import grid_online
from precycle.utils.utils import REPO_ROOT

app = typer.Typer()


tasks_path_prefix = REPO_ROOT.joinpath("data/covid19/covid19_workload/")
blocks_path_prefix = REPO_ROOT.joinpath("data/covid19/covid19_data/")
blocks_metadata = str(blocks_path_prefix.joinpath("blocks/metadata.json"))
blocks_path = str(blocks_path_prefix.joinpath("blocks"))


def caching_monoblock():
    task_paths = ["1:1blocks_34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir="monoblock",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        # cache=["HybridCache"],
        cache=["HybridCache", "LaplaceCache", "PMWCache"],
        initial_blocks=[1],
        max_blocks=[1],
        avg_num_tasks_per_block=[25e3],
        max_tasks=[25e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0, 0.5, 1, 1.5],
        # zipf_k=[1.5],
        heuristic=[
            # "",
            # # "bin_visits:1-1",
            # "bin_visits:1-20",
            "bin_visits:100-10",        # Best
            # "bin_visits:200-10",
            # "bin_visits:300-20",
            # "bin_visits:500-20",
            # "bin_visits:1000-5",
            # "bin_visits:1000-20",
            # "bin_visits:1000-10",
        ],
        variance_reduction=[True],
        log_every_n_tasks=100,
        learning_rate=[0.2], #[0.05, 0.1, 0.15, 0.2],
        bootstrapping=[False],

    )


def caching_static_multiblock_laplace_vs_hybrid():
    task_paths = ["1:2:4:8:16:32:64:128blocks_34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir="static-multiblock_laplace_vs_hybrid",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        cache=["LaplaceCache", "HybridCache"],
        initial_blocks=[150],
        max_blocks=[150],
        block_selection_policy=["RandomBlocks"],
        avg_num_tasks_per_block=[2e3],
        max_tasks=[300e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0, 0.5, 1, 1.5],
        heuristic=[
            "bin_visits:100-5",
        ],
        variance_reduction=[True],
        log_every_n_tasks=500,
        learning_rate = [0.2],
        bootstrapping=[False]
    )

def caching_static_multiblock_heuristics():
    task_paths = ["1:2:4:8:16:32:64:128blocks_34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir="static-multiblock_heuristics",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        cache=["HybridCache"],
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
        learning_rate = [0.2],
        bootstrapping=[False]
    )

def caching_static_multiblock_learning_rate():
    task_paths = ["1:2:4:8:16:32:64:128blocks_34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir="static-multiblock_learning_rate",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        cache=["HybridCache"],
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
            "bin_visits:100-5",
        ],
        variance_reduction=[True],
        log_every_n_tasks=500,
        learning_rate = [0.05, 0.1, 0.15, 0.2],
        bootstrapping=[False]
    )


def caching_dynamic_multiblock():
    # task_paths = ["1:1:1:1:1:2:5:10:20:30:60:90blocks_34425queries.privacy_tasks.csv"]
    task_paths = ["1:1:1:2:5:10blocks_34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir="multiexps/dynamic_multiblock_checking_new_renyi200",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        cache=["HybridCache"],
        # cache=["LaplaceCache", "HybridCache"],
        initial_blocks=[1],
        # max_blocks=[150],
        max_blocks=[20],
        block_selection_policy=["LatestBlocks"],
        # avg_num_tasks_per_block=[5e4],
        avg_num_tasks_per_block=[10e3],
        # avg_num_tasks_per_block=[25e2],
        # max_tasks=[75e5],
        max_tasks=[200e3],
        # max_tasks=[50e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0.5],
        # zipf_k=[0, 0.5, 1, 1.5],
        heuristic=[
            # "",
            # "bin_visits:500-10",
            # "bin_visits:100-5",
            # "bin_visits:200-5",
            # "bin_visits:200-5",
            "bin_visits:500-20",
            # "bin_visits:1000-1",
            # "bin_visits:500-50",
            # "bin_visits:200-20",
            # "bin_visits:200-10",
            # "bin_visits:100-20",
            # "bin_visits:200-50",
        ],
        variance_reduction=[True],
        log_every_n_tasks=500,
        learning_rate=[0.05, 0.1, 0.15, 0.2],
        # learning_rate = [0.1, 0.15, 0.2],
        bootstrapping=[True, False],
    )


@app.command()
def run(exp: str = "caching_monoblock", loguru_level: str = "INFO"):

    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}"]()


if __name__ == "__main__":
    app()
