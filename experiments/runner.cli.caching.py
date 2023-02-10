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
        cache=["CombinedCache", "DeterministicCache", "ProbabilisticCache"],
        # cache=["DeterministicCache", "CombinedCache"],
        initial_blocks=[1],
        max_blocks=[1],
        avg_num_tasks_per_block=[15e3],
        max_tasks=[15e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.0001],
        zipf_k=[0, 0.5, 1, 1.5],
        heuristic="total_updates_counts",
        heuristic_value=[1000],  # [100, 250, 500, 750, 1000]
        max_pmw_k=[1],
        variance_reduction=[True],
    )


def caching_multiblock():
    task_paths = ["1:1:1:2:4:8:16:32blocks_34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        global_seed=64,
        logs_dir="experiment",
        tasks_path=task_paths,
        blocks_path=blocks_path,
        blocks_metadata=blocks_metadata,
        planner=["MinCuts"],
        # cache=["CombinedCache", "DeterministicCache", "ProbabilisticCache"],
        cache=["CombinedCache"],
        initial_blocks=[1],
        max_blocks=[100],
        avg_num_tasks_per_block=[15e1],
        max_tasks=[15e3],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.0001],
        # zipf_k=[0, 0.5, 1, 1.5],
        zipf_k=[0.5],
        heuristic="total_updates_counts",
        heuristic_value=[100],  # [100, 250, 500, 750, 1000]
        max_pmw_k=[5],
        variance_reduction=[True],
    )


@app.command()
def run(exp: str = "caching_monoblock", loguru_level: str = "INFO"):

    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}"]()


if __name__ == "__main__":
    app()
