import os
from pathlib import Path
import typer

from experiments.ray_runner import (
    grid_online,
)

app = typer.Typer()


def budget_utilization():
    grid_online(
        scheduler_scheduling_time=[1],
        metric_recomputation_period=[50],
        initial_blocks=[1],
        max_blocks=[400],
        tasks_data_path=["covid19/privacy_tasks.csv"],
        blocks_data_path=["covid19/blocks"],
        tasks_sampling="",
        data_lifetime=[0.1],
        k=[0.25],
        allow_block_substitution=True,
    )


@app.command()
def run(
    exp: str = "budget_utilization",
    loguru_level: str = "WARNING",
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    globals()[f"{exp}"]()


if __name__ == "__main__":
    app()
