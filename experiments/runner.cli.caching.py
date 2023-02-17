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
        logs_dir="experiment",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        # cache=["ProbabilisticCache"],
        cache=["CombinedCache", "DeterministicCache", "ProbabilisticCache"],
        initial_blocks=[1],
        max_blocks=[1],
        avg_num_tasks_per_block=[1e5],
        # max_tasks=[1e5],
        max_tasks=[1e3],
        block_selection_policy=["RandomBlocks"],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.0001],
        # zipf_k=[0, 0.5, 1, 1.5],
        zipf_k=[0],
        heuristic=[
            # "bin_visits:1-5",
            "bin_visits:1-10",
            # "bin_visits:1-15",
            # "bin_visits:1-20",
            # "bin_visits:1-30",
            # "bin_visits:1-40",
            # "bin_visits:1-50",
            # "bin_visits:1-100",
            # "bin_visits:100-5",
            # "bin_visits:100-10",
            # "bin_visits:100-15",
            # "bin_visits:100-20",
            # "total_updates_counts:100",
        ],
        max_pmw_k=[1],  # It's important to be 1 in the monoblock case
        variance_reduction=[True],
    )


def caching_static_multiblock():
    # task_paths = ["1:1:2:7:7:14:30:60:90:120:150:180:210:240:blocks_34425queries.privacy_tasks.csv"]
    # task_paths = ["1:2:7:14:21:30:60:90:120blocks_34425queries.privacy_tasks.csv"]
    task_paths = ["1:2:4blocks_34425queries.privacy_tasks.csv"]

    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=8,
        logs_dir="experiment",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        # cache=["CombinedCache", "DeterministicCache", "ProbabilisticCache"],
        # cache=["DeterministicCache"],
        cache=["CombinedCache", "DeterministicCache"],
        initial_blocks=[5],
        max_blocks=[5],
        avg_num_tasks_per_block=[20000],
        max_tasks=[100000],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.001],
        zipf_k=[0.5],
        # zipf_k=[0.0],
        heuristic=[
            "bin_visits:1000-50",
            # "total_updates_counts:100",
        ],
        max_pmw_k=[2],
        variance_reduction=[True],
    )


@app.command()
def run(exp: str = "caching_monoblock", loguru_level: str = "INFO"):

    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}"]()


if __name__ == "__main__":
    app()
