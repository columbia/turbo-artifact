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
        global_seed=4,
        logs_dir="monoblock",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        cache=["CombinedCache"],
        # cache=["DeterministicCache"],
        # cache=["CombinedCache", "DeterministicCache"],  # , "ProbabilisticCache"],
        initial_blocks=[1],
        max_blocks=[1],
        avg_num_tasks_per_block=[5e4],
        max_tasks=[5e4],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0, 0.5, 1, 1.5],
        # zipf_k=[1],
        heuristic=[
            # "",
            # "bin_visits:1-1",
            # "bin_visits:1-20",
            "bin_visits:500-20",
            # "bin_visits:1000-20",
            # "bin_visits:1-100",
            # "bin_visits:500-100",
            # "bin_visits:1000-100",
        ],
        variance_reduction=[True],
        log_every_n_tasks=100,
    )


def caching_static_multiblock():
    # task_paths = ["1:1:2:7:7:14:30:60:90:120:150:180:210:240:blocks_34425queries.privacy_tasks.csv"]
    # task_paths = ["1:2:7:14:21:30:60:90:120blocks_34425queries.privacy_tasks.csv"]
    task_paths = ["1:2:4blocks_34425queries.privacy_tasks.csv"]

    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=128,
        logs_dir="static_multiblock_renyi",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        # cache=["CombinedCache"],
        cache=["DeterministicCache"],
        # cache=["CombinedCache", "DeterministicCache"],
        initial_blocks=[5],
        max_blocks=[5],
        block_selection_policy=["RandomBlocks"],
        avg_num_tasks_per_block=[20e3],
        max_tasks=[100e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0.5],
        # zipf_k=[0, 0.5, 1, 1.5],
        heuristic=[
            "",
            # "bin_visits:500-50",
            # "bin_visits:500-10",
            # "bin_visits:200-10",
            # "bin_visits:400-50",
            # "bin_visits:1000-50",
            # "bin_visits:1000-20",
            # "bin_visits:1000-50",
            # "bin_visits:1000-100",
            # "total_updates_counts:100",
        ],
        variance_reduction=[True],
        log_every_n_tasks=500,
        # learning_rate = [0.05, 0.1, 0.15, 0.2] #[0.05, 0.01, 0.1]
    )


def caching_dynamic_multiblock():
    # task_paths = ["1:1:1:1:1:2:5:10:20:30:60:90blocks_34425queries.privacy_tasks.csv"]
    task_paths = ["1:1:1:2:5:10blocks_34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir="multiexps/dynamic_multiblock_no_bootstrap_renyi3",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        cache=["CombinedCache"],
        # cache=["DeterministicCache"],
        # cache=["CombinedCache", "DeterministicCache"],
        initial_blocks=[1],
        # max_blocks=[150],
        max_blocks=[20],
        block_selection_policy=["LatestBlocks"],
        # avg_num_tasks_per_block=[5e4],
        avg_num_tasks_per_block=[10e3],
        # max_tasks=[75e5],
        max_tasks=[200e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0.5],
        # zipf_k=[0, 0.5, 1, 1.5],
        heuristic=[
            # "",
            "bin_visits:500-10",
            "bin_visits:500-20",
            "bin_visits:500-50",
        ],
        variance_reduction=[True],
        log_every_n_tasks=500,
        learning_rate = [0.05, 0.1, 0.15, 0.2],
        bootstrapping = [True, False]
    )


@app.command()
def run(exp: str = "caching_monoblock", loguru_level: str = "INFO"):

    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}"]()


if __name__ == "__main__":
    app()
