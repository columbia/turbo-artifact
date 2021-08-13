from pathlib import Path
from typing import Iterable, Tuple

import yaml

from privacypacking.budget import Budget

EPSILON = "epsilon"
DELTA = "delta"
BLOCKS_SPEC = "blocks_spec"
TASKS_SPEC = "tasks_spec"
OFFLINE = "offline"
ONLINE = "online"
ENABLED = "enabled"
NUM = "num"
CURVE_DISTRIBUTIONS = "curve_distributions"
LAPLACE = "laplace"
GAUSSIAN = "gaussian"
SUBSAMPLEGAUSSIAN = "SubsampledGaussian"
NOISE_START = "noise_start"
NOISE_STOP = "noise_stop"
SIGMA_START = "sigma_start"
SIGMA_STOP = "sigma_stop"
DATASET_SIZE = "dataset_size"
BATCH_SIZE = "batch_size"
EPOCHS = "epochs"
SCHEDULER_SPEC = "scheduler_spec"
SIMPLEX = "simplex"
OFFLINE_DPF = "offline_dpf"
FCFS = "fcfs"
DPF = "dpf"
NAME = "name"
N = "n"
PLOT_FILE = "plot_file"
LOG_FILE = "log_file"
FREQUENCY = "frequency"
TASK_ARRIVAL_FREQUENCY = "task_arrival_frequency"
BLOCK_ARRIVAL_FRQUENCY = "block_arrival_frequency"

GLOBAL_SEED = "global_seed"
DETERMINISTIC = "deterministic"
RANDOM = "random"
POISSON = "poisson"
CONSTANT = "constant"
TASK_ARRIVAL_INTERVAL = "task_arrival_interval"
BLOCK_ARRIVAL_INTERVAL = "block_arrival_interval"
BLOCKS_REQUEST = "blocks_request"
BLOCKS_NUM_MAX = "blocks_num_max"
BLOCKS_NUM = "blocks_num"
BLOCK_SELECTING_POLICY = "block_selecting_policy"
LATEST_FIRST = "latest_first"

PRIVATEKUBE_DEMANDS_PATH = Path(__file__).parent.parent.parent.joinpath(
    "data/privatekube_demands"
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


def load_blocks_and_budgets_from_dir(
    path: Path = PRIVATEKUBE_DEMANDS_PATH,
) -> Iterable[Tuple[int, Budget]]:
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
