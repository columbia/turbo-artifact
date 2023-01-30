import os
import typer

from experiments.ray_runner import grid_online
from precycle.utils.utils import REPO_ROOT

app = typer.Typer()


def caching():
    tasks_path_prefix = REPO_ROOT.joinpath("data/covid19/covid19_workload/")
    blocks_path_prefix = REPO_ROOT.joinpath("data/covid19/covid19_data/")

    task_paths = ["1:1blocks_34425queries.privacy_tasks.csv"]
    task_paths = [
        str(tasks_path_prefix.joinpath(task_path)) for task_path in task_paths
    ]

    grid_online(
        logs_dir="experiment",
        tasks_path=task_paths,
        blocks_path=str(blocks_path_prefix.joinpath("blocks")),
        blocks_metadata=str(blocks_path_prefix.joinpath("blocks/metadata.json")),
        planner=["MinCutsPlanner"],
        cache=["CombinedCache", "DeterministicCache", "ProbabilisticCache"],
        # cache=["CombinedCache"],
        initial_blocks=[1],
        max_blocks=[1],
        avg_num_tasks_per_block=[5e4],
        max_tasks=[5e4],
        initial_tasks=[0],
        alpha=[0.05],
        beta=[0.0001],
        zipf_k=[0, 0.5, 1, 1.5],
        heuristic="total_updates_counts",
        heuristic_value=[2000], #[100, 250, 500, 750, 1000]
    )


@app.command()
def run(exp: str = "caching", loguru_level: str = "CRITICAL"):

    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    globals()[f"{exp}"]()


if __name__ == "__main__":
    app()
