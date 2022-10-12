import json
from collections import namedtuple
from pathlib import Path

from omegaconf import OmegaConf

from privacypacking.schedulers.utils import ALLOCATED
import pandas as pd

CUSTOM_LOG_PREFIX = "custom_log_prefix"
REPO_ROOT = Path(__file__).parent.parent.parent
LOGS_PATH = REPO_ROOT.joinpath("logs")
RAY_LOGS = LOGS_PATH.joinpath("ray")
DEFAULT_CONFIG_FILE = REPO_ROOT.joinpath("privacypacking/config/default.yaml")

TaskSpec = namedtuple(
    "TaskSpec", ["profit", "block_selection_policy", "n_blocks", "budget", "name"]
)


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

        alternative_plan_result = None
        if task.id in tasks_info.alternative_plan_result:
            alternative_plan_result = tasks_info.alternative_plan_result[task.id]

        original_plan_result = None
        if task.id in tasks_info.original_plan_result:
            original_plan_result = tasks_info.original_plan_result[task.id]

        task_dump.update(
            {
                "allocated": tasks_info.tasks_status[task.id] == ALLOCATED,
                "status": tasks_info.tasks_status[task.id],
                "original_plan_result": original_plan_result,
                "alternative_plan_result": alternative_plan_result,
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

    dfs = []
    log_blocks = []
    for block in blocks.values():
        log_blocks.append(block.dump())
        dfs.append(pd.DataFrame([{"budget": block.budget.epsilon}]))
    df = pd.concat(dfs)

    df['budget'] = 10 - df['budget']
    bu = df['budget'].mean()

    total_tasks = len(tasks)
    # tasks_info = tasks_info.dump()
    # tasks_scheduling_times = sorted(tasks_scheduling_times)
    allocated_tasks_scheduling_delays = allocated_tasks_scheduling_delays
    simulator_config = simulator_config.dump()
    omegaconf = OmegaConf.create(simulator_config["omegaconf"])

    datapoint = {
        "scheduler": omegaconf.scheduler.method,
        "solver": omegaconf.scheduler.solver,
        "scheduler_n": omegaconf.scheduler.n,
        "scheduler_metric": omegaconf.scheduler.metric,
        "T": omegaconf.scheduler.scheduling_wait_time,
        "budget_utilization": bu,
        "realized_budget": tasks_info.realized_budget,
        "data_lifetime": omegaconf.scheduler.data_lifetime,
        "block_selecting_policy": omegaconf.tasks.block_selection_policy,
        "n_allocated_tasks": n_allocated_tasks,
        "original_plans_ran": tasks_info.original_plans_ran,
        "alternative_plans_ran": tasks_info.alternative_plans_ran,
        "max_aggregations_allowed": omegaconf.scheduler.max_aggregations_allowed,
        "total_tasks": total_tasks,
        "realized_profit": realized_profit,
        "n_initial_blocks": omegaconf.blocks.initial_num,
        "maximum_profit": maximum_profit,
        "mean_task_per_block": omegaconf.tasks.avg_num_tasks_per_block,
        "data_path": omegaconf.tasks.data_path,
        "allocated_tasks_scheduling_delays": allocated_tasks_scheduling_delays,
        "initial_blocks": omegaconf.blocks.initial_num,
        "max_blocks": omegaconf.blocks.max_num,
        "tasks": log_tasks,
        "blocks": log_blocks,
        "task_frequencies_path": omegaconf.tasks.frequencies_path,
        "tasks_path": omegaconf.tasks.tasks_path,
        "metric_recomputation_period": omegaconf.scheduler.metric_recomputation_period,
        "normalize_by": omegaconf.metric.normalize_by,
        "temperature": omegaconf.metric.temperature,
    }

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
