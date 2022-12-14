import os
import typer

from experiments.ray_runner import (
    grid_online,
)

from privacypacking.schedulers.utils import FCFS

app = typer.Typer()


def caching():
    grid_online(
        scheduler_scheduling_time=[1],
        metric_recomputation_period=[50],
        scheduler_metrics=[FCFS],
        n=[1],  # Instant unlocking
        max_blocks=[500],
        initial_blocks=[0],
        initial_tasks=[0],
        logs_dir="2objexp1",
        tasks_path=["covid19/covid19_workload/privacy_tasks.csv"],
        queries_path=["covid19/covid19_queries/queries.json"],
        blocks_path="covid19/covid19_data/blocks",
        blocks_metadata="covid19/covid19_data/metadata.json",
        tasks_sampling="",
        data_lifetime=[0.1],
        task_lifetime=[1],
        # planner=["DynamicProgrammingPlanner:2", "PerBlockPlanner", "NoPlanner"],
        # planner=["DynamicProgrammingPlannerUtility:2"],
        planner=["ILP"],
        optimization_objective=["minimize_budget", "minimize_aggregations"],  # ["minimize_budget"]
        variance_reduction=[False, True],
        cache=["DeterministicCache"],  # ProbabilisticCache
        enable_caching=[True],
        enable_dp=[True],
        repetitions=1,
        enable_random_seed=True,
        utility=[50, 100, 200, 500, 1000],
    )


@app.command()
def run(
    exp: str = "caching",
    loguru_level: str = "WARNING",
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    globals()[f"{exp}"]()


if __name__ == "__main__":
    app()