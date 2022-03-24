import json
from collections import namedtuple
from pathlib import Path

from omegaconf import OmegaConf

from privacypacking.schedulers.utils import ALLOCATED

EPSILON = "epsilon"
DELTA = "delta"
BLOCKS_SPEC = "blocks_spec"
TASKS_SPEC = "tasks_spec"
SCHEDULER_SPEC = "scheduler_spec"
ENABLED = "enabled"
NUM = "num"
CURVE_DISTRIBUTIONS = "curve_distributions"
CUSTOM = "custom"
DATA_PATH = "data_path"
DATA_TASK_FREQUENCIES_PATH = "data_task_frequencies_path"
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
# MAX_TASKS = "max_tasks"
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


def get_logs(
    tasks,
    blocks,
    tasks_info,
    simulator_config,
    **kwargs,
) -> dict:
    n_allocated_tasks = 0
    tasks_scheduling_times = []
    allocated_tasks_scheduling_delays = []
    maximum_profit = 0
    realized_profit = 0

    log_tasks = []
    for task in tasks:
        task_dump = task.dump()

        maximum_profit += task.profit
        if tasks_info.tasks_status[task.id] == ALLOCATED:
            n_allocated_tasks += 1
            realized_profit += task.profit
            tasks_scheduling_times.append(tasks_info.scheduling_time[task.id])
            allocated_tasks_scheduling_delays.append(
                tasks_info.scheduling_delay.get(task.id, None)
            )

        task_dump.update(
            {
                "allocated": tasks_info.tasks_status[task.id] == ALLOCATED,
                "status": tasks_info.tasks_status[task.id],
                "creation_time": tasks_info.creation_time[task.id],
                "scheduling_time": tasks_info.scheduling_time.get(task.id, None),
                "scheduling_delay": tasks_info.scheduling_delay.get(task.id, None),
                "allocation_index": tasks_info.allocation_index.get(task.id, None),
            }
        )
        log_tasks.append(
            task_dump
        )  # todo change allocated_task_ids from list to a set or sth more efficient for lookups

    # TODO: Store scheduling times into the tasks directly?

    log_blocks = []
    for block in blocks.values():
        log_blocks.append(block.dump())

    n_allocated_tasks = n_allocated_tasks
    total_tasks = len(tasks)
    # tasks_info = tasks_info.dump()
    # tasks_scheduling_times = sorted(tasks_scheduling_times)
    allocated_tasks_scheduling_delays = allocated_tasks_scheduling_delays
    simulator_config = simulator_config.dump()
    omegaconf = OmegaConf.create(simulator_config["omegaconf"])

    datapoint = {
        # "scheduler": simulator_config["scheduler_spec"]["method"],
        # "solver": simulator_config["scheduler_spec"]["solver"],
        # "scheduler_n": simulator_config["scheduler_spec"]["n"],
        # "scheduler_metric": simulator_config["scheduler_spec"]["metric"],
        "scheduler": omegaconf.scheduler.method,
        "solver": omegaconf.scheduler.solver,
        "scheduler_n": omegaconf.scheduler.n,
        "scheduler_metric": omegaconf.scheduler.metric,
        "T": omegaconf.scheduler.scheduling_wait_time,
        "data_lifetime": omegaconf.scheduler.data_lifetime,
        "block_selecting_policy": simulator_config["tasks_spec"]["curve_distributions"][
            "custom"
        ]["read_block_selecting_policy_from_config"]["block_selecting_policy"],
        "frequency_file": simulator_config["tasks_spec"]["curve_distributions"][
            "custom"
        ]["data_task_frequencies_path"],
        "n_allocated_tasks": n_allocated_tasks,
        "total_tasks": total_tasks,
        "realized_profit": realized_profit,
        "n_initial_blocks": omegaconf.blocks.initial_num,
        "maximum_profit": maximum_profit,
        "mean_task_per_block": simulator_config["tasks_spec"][TASK_ARRIVAL_FREQUENCY][
            POISSON
        ].get(AVG_NUMBER_TASKS_PER_BLOCK, None),
        "data_path": simulator_config["tasks_spec"]["curve_distributions"]["custom"][
            "data_path"
        ],
        "allocated_tasks_scheduling_delays": allocated_tasks_scheduling_delays,
        "tasks": log_tasks,
        "blocks": log_blocks,
    }

    datapoint[
        "metric_recomputation_period"
    ] = omegaconf.scheduler.metric_recomputation_period
    datapoint["normalize_by"] = omegaconf.metric.normalize_by
    datapoint["temperature"] = omegaconf.metric.temperature
    # TODO: remove allocating_task_id from args
    # TODO: Store scheduling times into the tasks directly?

    # Any other thing to log
    for key, value in kwargs.items():
        datapoint[key] = value
    return datapoint


def save_logs(config, log_dict, compact=False, compressed=False):
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        raise NotImplementedError
    else:
        with open(config.log_path, "w") as fp:
            if compact:
                json_object = json.dumps(log_dict, separators=(",", ":"))
            else:
                json_object = json.dumps(log_dict, indent=4)

            fp.write(json_object)


def global_metrics(logs: dict, verbose=False) -> dict:
    if not verbose:
        logs["tasks"] = ""
        logs["blocks"] = ""
    return logs
