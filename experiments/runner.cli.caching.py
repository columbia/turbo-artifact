import os
import typer

from experiments.ray_runner import grid_online

from privacypacking.schedulers.utils import FCFS

app = typer.Typer()


def caching():
    grid_online(
        scheduler_scheduling_time=[1],
        metric_recomputation_period=[50],
        scheduler_metrics=[FCFS],
        n=[1],  # Instant unlocking
        logs_dir="experiment",
        # tasks_path=["covid19/covid19_workload/privacy_tasks.csv"],
        # queries_path=["covid19/covid19_queries/queries.json"],
        tasks_path=["covid19/covid19_workload/400:7blocks_84queries.privacy_tasks.csv"],
        queries_path=["covid19/covid19_queries/all_2way_marginals.queries.json"],
        blocks_path="covid19/covid19_data/blocks",
        blocks_metadata="covid19/covid19_data/metadata.json",
        max_blocks=[400],
        initial_blocks=[1],
        data_lifetime=[0.1],
        initial_tasks=[0],
        avg_num_tasks_per_block=[100],
        # max_tasks=[4000],
        task_lifetime=[1],
        # planner=["MinCutsPlanner"],
        planner=["MaxCutsPlanner"],
        # planner=["ILP"],
        # optimization_objective=["minimize_budget"],   # Disabled - objective fixed to minimize budget
        variance_reduction=[False],  # [True, False],
        cache=["DeterministicCache"],  # ProbabilisticCache
        enable_caching=[True],
        enable_dp=[True],
        repetitions=1,
        enable_random_seed=True,
        alpha=[0.005],
        beta=[0.0001],
    )


@app.command()
def run(
    exp: str = "caching",
    loguru_level: str = "INFO",
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    globals()[f"{exp}"]()


if __name__ == "__main__":
    app()
