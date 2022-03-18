import argparse
import cProfile
import os
import pstats
from datetime import datetime
from functools import partial
from typing import Any, Dict, List

import ray
import typer
import yaml

# os.environ["LOGURU_LEVEL"] = "DEBUG"
os.environ["LOGURU_LEVEL"] = "WARNING"


from loguru import logger
from ray import tune

from privacypacking import schedulers
from privacypacking.config import Config
from privacypacking.schedulers.utils import (
    ARGMAX_KNAPSACK,
    BASIC_SCHEDULER,
    BATCH_OVERFLOW_RELEVANCE,
    DOMINANT_SHARES,
    DYNAMIC_FLAT_RELEVANCE,
    FCFS,
    FLAT_RELEVANCE,
    NAIVE_AVERAGE,
    OVERFLOW_RELEVANCE,
    SIMPLEX,
    SOFT_KNAPSACK,
    THRESHOLD_UPDATING,
    TIME_BASED_BUDGET_UNLOCKING,
    VECTORIZED_BATCH_OVERFLOW_RELEVANCE,
)
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import *

app = typer.Typer()


@app.command()
def main(
    loguru_level: str = "WARNING",
):

    profiler = cProfile.Profile()
    profiler.enable()

    run_online(
        custom_config="time_based_budget_unlocking/privatekube/base.yaml",
        initial_blocks=10,
        max_blocks=20,
        avg_number_tasks_per_block=500,
        data_path="mixed_curves",
        metric_recomputation_period=100,
        parallel=False,  # We care about the runtime here
    )
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("ncalls")
    stats.dump_stats("logs/out.prof")


def run_online(
    custom_config: str,
    avg_number_tasks_per_block: int,
    initial_blocks: int,
    max_blocks: int,
    data_path: str = "",
    optimal: bool = False,
    metric_recomputation_period: int = 10,
    parallel: bool = True,
):
    with open(DEFAULT_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    with open(
        DEFAULT_CONFIG_FILE.parent.joinpath(custom_config),
        "r",
    ) as user_config:
        user_config = yaml.safe_load(user_config)
    update_dict(user_config, config)

    temperature = -1

    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM].update(
        {
            # SAMPLING: True,
            # INITIAL_NUM: num_tasks,
            DATA_PATH: data_path,
            DATA_TASK_FREQUENCIES_PATH: "frequencies.yaml",
            FREQUENCY: 1,
            READ_BLOCK_SELECTION_POLICY_FROM_CONFIG: {
                ENABLED: True,
                BLOCK_SELECTING_POLICY: "LatestBlocksFirst",
            },
        }
    )

    config[TASKS_SPEC][TASK_ARRIVAL_FREQUENCY][POISSON][
        AVG_NUMBER_TASKS_PER_BLOCK
    ] = avg_number_tasks_per_block

    config["omegaconf"] = {
        "scheduler": {
            "method": "batch",
            "metric": ARGMAX_KNAPSACK,
            "metric_recomputation_period": metric_recomputation_period,
            "log_warning_every_n_allocated_tasks": 50,
            "scheduler_timeout_seconds": 20 * 60,
        },
        "metric": {
            "normalize_by": "available_budget",
            "temperature": temperature,
            "n_knapsack_solvers": 16 if parallel else 1,
            "gurobi_timeout": 10 * 60,
            "gurobi_threads": 8 if parallel else 1,
        },
        "logs": {
            "verbose": False,
            "save": True,
        },
        "blocks": {
            "initial_num": initial_blocks,
            "max_num": max_blocks,
        },
    }

    sim = Simulator(Config(config))

    metrics = sim.run()

    logger.warning(
        f"Done. Result: allocated {metrics['n_allocated_tasks']}/{metrics['total_tasks']} tasks."
    )


if __name__ == "__main__":
    app()
