import json
from collections import namedtuple
from pathlib import Path
from typing import Iterable, Tuple

import yaml

from privacypacking.budget import Budget
from privacypacking.budget.block_selection import BlockSelectionPolicy

EPSILON = "epsilon"
DELTA = "delta"
BLOCKS_SPEC = "blocks_spec"
TASKS_SPEC = "tasks_spec"
SCHEDULER_SPEC = "scheduler_spec"
OFFLINE = "offline"
ONLINE = "online"
ENABLED = "enabled"
NUM = "num"
CURVE_DISTRIBUTIONS = "curve_distributions"
LAPLACE = "laplace"
GAUSSIAN = "gaussian"
SUBSAMPLEGAUSSIAN = "SubsampledGaussian"
CUSTOM = "custom"
DATA_PATH = "data_path"
NOISE_START = "noise_start"
NOISE_STOP = "noise_stop"
SIGMA_START = "sigma_start"
SIGMA_STOP = "sigma_stop"
DATASET_SIZE = "dataset_size"
BATCH_SIZE = "batch_size"
EPOCHS = "epochs"

NAME = "name"
N = "n"
PLOT_FILE = "plot_file"
LOG_FILE = "log_file"
FREQUENCY = "frequency"
TASK_ARRIVAL_FREQUENCY = "task_arrival_frequency"
BLOCK_ARRIVAL_FRQUENCY = "block_arrival_frequency"

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

# Block selecting policies
BLOCK_SELECTING_POLICY = "block_selecting_policy"
LATEST_BLOLCKS_FIRST = "latest_blocks_first"
RANDOM_BLOCKS = "random_blocks"

# Schedulers
SIMPLEX = "simplex"
OFFLINE_DPF = "offline_dpf"
FCFS = "fcfs"
DPF = "dpf"

REPO_ROOT = Path(__file__).parent.parent.parent

PRIVATEKUBE_DEMANDS_PATH = REPO_ROOT.joinpath("data/privatekube_demands")
LOGS_PATH = REPO_ROOT.joinpath("logs")
DEFAULT_CONFIG_FILE = REPO_ROOT.joinpath("privacypacking/config/default_config.yaml")
RAY_LOGS = LOGS_PATH.joinpath("ray")

TaskParameters = namedtuple(
    "TaskDistribution", ["n_blocks", "policy", "profit", "budget"]
)

TaskSpec = namedtuple(
    "TaskSpec", ["profit", "block_selection_policy", "n_blocks", "budget"]
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
        "scheduler": logs["simulator_config"]["scheduler_spec"]["name"],
        "n_allocated_tasks": n_allocated_tasks,
        "realized_profit": realized_profit,
        "n_tasks": n_tasks,
        "maximum_profit": maximum_profit,
        "scheduling_time": logs["scheduling_time"],
    }

    return datapoint


def load_blocks_and_budgets_from_dir(
    path: Path = PRIVATEKUBE_DEMANDS_PATH,
) -> Iterable[Tuple[int, "Budget"]]:
    blocks_and_budgets = []
    block_rescaling_factor = 100  # The logs have 100 blocks per day
    for yaml_path in path.glob("**/*.yaml"):
        with open(yaml_path, "r") as f:
            demand_dict = yaml.safe_load(f)
            # print(demand_dict)
            orders = {}
            for i, alpha in enumerate(demand_dict["alphas"]):
                orders[alpha] = demand_dict["rdp_epsilons"][i]
            blocks_and_budgets.append(
                (demand_dict["n_blocks"] // block_rescaling_factor, Budget(orders))
            )
    return blocks_and_budgets


# todo: why we may have more than one yaml files?
def load_task_distribution(
    path: Path = PRIVATEKUBE_DEMANDS_PATH,
) -> Iterable[TaskParameters]:
    task_distribution = []
    for yaml_path in path.glob("**/*.yaml"):
        with open(yaml_path, "r") as f:
            demand_dict = yaml.safe_load(f)
            orders = {}
            for i, alpha in enumerate(demand_dict["alphas"]):
                orders[alpha] = demand_dict["rdp_epsilons"][i]

            if "policy" in demand_dict:
                policy = BlockSelectionPolicy.from_str(demand_dict["policy"])
            else:
                policy = BlockSelectionPolicy.from_str("RandomBlocks")

            profit = demand_dict.get("profit", 1)

            task_distribution.append(
                TaskParameters(
                    n_blocks=demand_dict["n_blocks"],
                    policy=policy,
                    profit=profit,
                    budget=Budget(orders),
                )
            )

    return task_distribution


def load_task_spec_from_file(path: Path = PRIVATEKUBE_DEMANDS_PATH) -> TaskSpec:
    with open(path, "r") as f:
        demand_dict = yaml.safe_load(f)
        orders = {}
        for i, alpha in enumerate(demand_dict["alphas"]):
            orders[alpha] = demand_dict["rdp_epsilons"][i]

        profit = demand_dict.get("profit", 1)
        if "block_selection_policy" in demand_dict:
            block_selection_policy = BlockSelectionPolicy.from_str(
                demand_dict["block_selection_policy"]
            )
        else:
            block_selection_policy = BlockSelectionPolicy.from_str("RandomBlocks")

        task_spec = TaskSpec(
            profit=profit,
            block_selection_policy=block_selection_policy,
            n_blocks=demand_dict["n_blocks"],
            budget=Budget(orders),
        )
    assert task_spec is not None
    return task_spec
