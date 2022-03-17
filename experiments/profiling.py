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
    # main()

    run_one(
        custom_config="offline_dpf_killer/multi_block/gap_base.yaml",
        num_blocks=20,
        num_tasks=1_000,
        data_path="mixed_curves",
        metric_recomputation_period=50,
        parallel=False,  # We care about the runtime here
    )
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("ncalls")
    stats.dump_stats("logs/out.prof")


def run_one(
    custom_config: str,
    num_tasks: int,
    num_blocks: int,
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

    # Conditonal parameter
    method_and_metric = []
    for metric in [
        # DOMINANT_SHARES,
        # FLAT_RELEVANCE,
        # OVERFLOW_RELEVANCE,
        ARGMAX_KNAPSACK,
    ]:
        method_and_metric.append((BASIC_SCHEDULER, metric))

    if optimal:
        method_and_metric.append((SIMPLEX, DOMINANT_SHARES))

    config["method_and_metric"] = method_and_metric[0]

    block_selection_policies = "RandomBlocks"

    # num_tasks = [50, 100, 150, 200]
    # num_tasks = [100]
    # num_blocks = [5, 10, 15, 20]
    # num_blocks = [5]

    temperature = -1

    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][INITIAL_NUM] =
    #     np.arange(1, 500, step=1, dtype=int).tolist()
    # )
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][INITIAL_NUM] =
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM].update(
        {
            SAMPLING: True,
            INITIAL_NUM: num_tasks,
            DATA_PATH: data_path,
            DATA_TASK_FREQUENCIES_PATH: "frequencies.yaml",
            FREQUENCY: 1,
            READ_BLOCK_SELECTION_POLICY_FROM_CONFIG: {
                ENABLED: True,
                BLOCK_SELECTING_POLICY: block_selection_policies,
            },
        }
    )
    config[BLOCKS_SPEC][INITIAL_NUM] = num_blocks

    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
    #     READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    # ][BLOCK_SELECTING_POLICY] = block_selection_policies)

    config[CUSTOM_LOG_PREFIX] = f"exp_{datetime.now().strftime('%m%d-%H%M%S')}"

    config["omegaconf"] = {
        "scheduler": {
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
    }

    # Unpack conditional parameters
    config[SCHEDULER_SPEC][METHOD], config[SCHEDULER_SPEC][METRIC] = config.pop(
        "method_and_metric"
    )

    sim = Simulator(Config(config))

    metrics = sim.run()

    logger.warning(
        f"Done. Result: allocated {metrics['n_allocated_tasks']}/{metrics['total_tasks']} tasks."
    )


if __name__ == "__main__":
    app()
