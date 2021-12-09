import json
import random
from collections import namedtuple
from pathlib import Path

import numpy as np
import yaml
from omegaconf import OmegaConf

from privacypacking.budget import Budget
from privacypacking.budget.block_selection import BlockSelectionPolicy

EPSILON = "epsilon"
DELTA = "delta"
BLOCKS_SPEC = "blocks_spec"
TASKS_SPEC = "tasks_spec"
SCHEDULER_SPEC = "scheduler_spec"
ENABLED = "enabled"
NUM = "num"
CURVE_DISTRIBUTIONS = "curve_distributions"
LAPLACE = "laplace"
GAUSSIAN = "gaussian"
SUBSAMPLEGAUSSIAN = "SubsampledGaussian"
CUSTOM = "custom"
DATA_PATH = "data_path"
DATA_TASK_FREQUENCIES_PATH = "data_task_frequencies_path"
NOISE_START = "noise_start"
NOISE_STOP = "noise_stop"
SIGMA_START = "sigma_start"
SIGMA_STOP = "sigma_stop"
DATASET_SIZE = "dataset_size"
BATCH_SIZE = "batch_size"
EPOCHS = "epochs"
READ_BLOCK_SELECTION_POLICY_FROM_CONFIG = "read_block_selecting_policy_from_config"
METHOD = "method"
METRIC = "metric"
N = "n"
DATA_LIFETIME = "data_lifetime"
PROFIT = "profit"
SOLVER = "solver"
GUROBI = "gurobi"
MIP = "mip"

NORMALIZE_BY_AVAILABLE_BUDGET = "normalize_by_available_budget"
NORMALIZE_BY_CAPACITY = "normalize_by_capacity"

THRESHOLD_UPDATE_MECHANISM = "threshold_update_mechanism"
PLOT_FILE = "plot_file"
LOG_FILE = "log_file"

FREQUENCY = "frequency"
TASK_ARRIVAL_FREQUENCY = "task_arrival_frequency"
AVG_NUMBER_TASKS_PER_BLOCK = "avg_number_tasks_per_block"
BLOCK_ARRIVAL_FRQUENCY = "block_arrival_frequency"
MAX_BLOCKS = "max_blocks"
MAX_TASKS = "max_tasks"
FROM_MAX_BLOCKS = "from_max_blocks"
GLOBAL_SEED = "global_seed"
DETERMINISTIC = "deterministic"
LOG_EVERY_N_ITERATIONS = "log_every_n_iterations"
RANDOM = "random"
POISSON = "poisson"
CONSTANT = "constant"
TASK_ARRIVAL_INTERVAL = "task_arrival_interval"
BLOCK_ARRIVAL_INTERVAL = "block_arrival_interval"
BLOCKS_REQUEST = "blocks_request"
NUM_BLOCKS_MAX = "num_blocks_max"
NUM_BLOCKS = "num_blocks"
INITIAL_NUM = "initial_num"
SAMPLING = "sampling"
BLOCK_SELECTING_POLICY = "block_selecting_policy"
SCHEDULING_WAIT_TIME = "scheduling_wait_time"
BUDGET_UNLOCKING_TIME = "budget_unlocking_time"
CUSTOM_LOG_PREFIX = "custom_log_prefix"

REPO_ROOT = Path(__file__).parent.parent.parent

PRIVATEKUBE_DEMANDS_PATH = REPO_ROOT.joinpath("data/privatekube_demands")
LOGS_PATH = REPO_ROOT.joinpath("logs")
DEFAULT_CONFIG_FILE = REPO_ROOT.joinpath("privacypacking/config/default_config.yaml")
RAY_LOGS = LOGS_PATH.joinpath("ray")

TaskSpec = namedtuple(
    "TaskSpec", ["profit", "block_selection_policy", "n_blocks", "budget", "name"]
)


def update_dict(src, des):
    ref = des
    for k, v in src.items():
        if isinstance(v, dict):
            prev_ref = ref
            ref = ref[k]
            update_dict(v, ref)
            ref = prev_ref
        else:
            ref[k] = v


def load_logs(log_path: str, relative_path=True) -> dict:
    full_path = Path(log_path)
    if relative_path:
        full_path = LOGS_PATH.joinpath(log_path)
    with open(full_path, "r") as f:
        logs = json.load(f)
    return logs


def global_metrics(logs: dict) -> dict:
    n_allocated_tasks = 0
    realized_profit = 0
    n_tasks = 0
    maximum_profit = 0

    for task in logs["tasks"]:
        n_tasks += 1
        maximum_profit += task["profit"]
        if task["allocated"]:
            n_allocated_tasks += 1
            realized_profit += task["profit"]

    datapoint = {
        "scheduler": logs["simulator_config"]["scheduler_spec"]["method"],
        "solver": logs["simulator_config"]["scheduler_spec"]["solver"],
        "scheduler_n": logs["simulator_config"]["scheduler_spec"]["n"],
        "scheduler_metric": logs["simulator_config"]["scheduler_spec"]["metric"],
        "block_selecting_policy": logs["simulator_config"]["tasks_spec"][
            "curve_distributions"
        ]["custom"]["read_block_selecting_policy_from_config"][
            "block_selecting_policy"
        ],
        "frequency_file": logs["simulator_config"]["tasks_spec"]["curve_distributions"][
            "custom"
        ]["data_task_frequencies_path"],
        "n_allocated_tasks": logs["num_scheduled_tasks"],
        "total_tasks": logs["total_tasks"],
        "realized_profit": realized_profit,
        "n_tasks": n_tasks,
        "n_initial_blocks": logs["simulator_config"]["blocks_spec"]["initial_num"],
        "maximum_profit": maximum_profit,
        "scheduling_time": logs["scheduling_time"],
        "T": logs["simulator_config"]["scheduler_spec"]["scheduling_wait_time"],
        "data_lifetime": logs["simulator_config"]["scheduler_spec"].get(
            "data_lifetime", None
        ),
        "mean_task_per_block": logs["simulator_config"]["tasks_spec"][
            TASK_ARRIVAL_FREQUENCY
        ][POISSON][AVG_NUMBER_TASKS_PER_BLOCK],
        "data_path": logs["simulator_config"]["tasks_spec"]["curve_distributions"][
            "custom"
        ]["data_path"],
        "allocated_tasks_scheduling_delays": logs["allocated_tasks_scheduling_delays"],
    }

    omegaconf = OmegaConf.create(logs["simulator_config"]["omegaconf"])
    datapoint[
        "metric_recomputation_period"
    ] = omegaconf.scheduler.metric_recomputation_period
    datapoint["normalize_by"] = omegaconf.metric.normalize_by
    datapoint["temperature"] = omegaconf.metric.temperature

    return datapoint
