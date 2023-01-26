import os
import typer

from experiments.ray_runner import grid_online
from precycle.utils.utils import REPO_ROOT

app = typer.Typer()


def caching():
    tasks_path_prefix = REPO_ROOT.joinpath("data/covid19/covid19_workload/")
    blocks_path_prefix = REPO_ROOT.joinpath("data/covid19/covid19_data/")

    # task_paths = [
    #     "1:1blocks_10queries.privacy_tasks.csv",
    #     "1:1blocks_100queries.privacy_tasks.csv",
    #     "1:1blocks_1000queries.privacy_tasks.csv",
    #     "1:1blocks_5000queries.privacy_tasks.csv",
    #     "1:1blocks_10000queries.privacy_tasks.csv",
    #     "1:1blocks_15000queries.privacy_tasks.csv",
    #     "1:1blocks_20000queries.privacy_tasks.csv",
    #     "1:1blocks_25000queries.privacy_tasks.csv",
    #     "1:1blocks_30000queries.privacy_tasks.csv",
    #     "1:1blocks_34000queries.privacy_tasks.csv",
    # ]
    task_paths = [
        "1:1blocks_34000queries.privacy_tasks.csv",
    ]

    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        logs_dir="experiment",
        tasks_path=task_paths,
        blocks_path=str(blocks_path_prefix.joinpath("blocks")),
        blocks_metadata=str(blocks_path_prefix.joinpath("metadata.json")),
        planner=["MinCutsPlanner"],
        cache=["CombinedCache"], #["DeterministicCache", "ProbabilisticCache"],
        initial_blocks=[1],
        max_blocks=[1],
        avg_num_tasks_per_block=[5e2],
        max_tasks=[5e2],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.0001],
        heuristic_threshold=[10, 50, 100, 1000],
    )


@app.command()
def run(exp: str = "caching", loguru_level: str = "INFO"):

    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}"]()


if __name__ == "__main__":
    app()
