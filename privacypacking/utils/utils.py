import yaml
import json
import numpy as np
from pathlib import Path
from collections import namedtuple
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
NOISE_START = "noise_start"
NOISE_STOP = "noise_stop"
SIGMA_START = "sigma_start"
SIGMA_STOP = "sigma_stop"
DATASET_SIZE = "dataset_size"
BATCH_SIZE = "batch_size"
EPOCHS = "epochs"

METHOD = "method"
METRIC = "metric"
N = "n"
PROFIT = "profit"

THRESHOLD_UPDATE_MECHANISM = "threshold_update_mechanism"
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
BLOCK_SELECTING_POLICY = "block_selecting_policy"

REPO_ROOT = Path(__file__).parent.parent.parent

PRIVATEKUBE_DEMANDS_PATH = REPO_ROOT.joinpath("data/privatekube_demands")
LOGS_PATH = REPO_ROOT.joinpath("logs")
DEFAULT_CONFIG_FILE = REPO_ROOT.joinpath("privacypacking/config/default_config.yaml")
RAY_LOGS = LOGS_PATH.joinpath("ray")

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
        "scheduler": logs["simulator_config"]["scheduler_spec"]["method"],
        "n_allocated_tasks": n_allocated_tasks,
        "realized_profit": realized_profit,
        "n_tasks": n_tasks,
        "maximum_profit": maximum_profit,
        "scheduling_time": logs["scheduling_time"],
    }

    return datapoint


def load_task_spec_from_file(path: Path = PRIVATEKUBE_DEMANDS_PATH) -> TaskSpec:
    with open(path, "r") as f:
        demand_dict = yaml.safe_load(f)
        orders = {}
        for i, alpha in enumerate(demand_dict["alphas"]):
            orders[alpha] = demand_dict["rdp_epsilons"][i]
        profit = demand_dict.get("profit", 1)
        block_selection_policy = None
        if "block_selection_policy" in demand_dict:
            block_selection_policy = BlockSelectionPolicy.from_str(
                demand_dict["block_selection_policy"]
            )
        assert block_selection_policy is not None

        n_blocks_requests = demand_dict["n_blocks"].split(",")
        num_blocks = [
            n_blocks_request.split(":")[0] for n_blocks_request in n_blocks_requests
        ]
        frequencies = [
            n_blocks_request.split(":")[1] for n_blocks_request in n_blocks_requests
        ]
        n_blocks = np.random.choice(
            num_blocks,
            1,
            p=frequencies,
        )[0]

        task_spec = TaskSpec(
            profit=profit,
            block_selection_policy=block_selection_policy,
            n_blocks=int(n_blocks),
            budget=Budget(orders),
        )
    assert task_spec is not None
    return task_spec
