import json
import uuid
import mlflow
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from privacypacking.schedulers.utils import ALLOCATED

CUSTOM_LOG_PREFIX = "custom_log_prefix"
REPO_ROOT = Path(__file__).parent.parent.parent
LOGS_PATH = REPO_ROOT.joinpath("logs")
RAY_LOGS = LOGS_PATH.joinpath("ray")
DEFAULT_CONFIG_FILE = REPO_ROOT.joinpath("privacypacking/config/default.yaml")

TaskSpec = namedtuple(
    "TaskSpec", ["profit", "block_selection_policy", "n_blocks", "budget", "name"]
)

import numpy as np


def mlflow_log(key, value, step):
    mlflow_run = mlflow.active_run()
    if mlflow_run:
        mlflow.log_metric(
            key,
            value,
            step=step,
        )


def sample_one_from_string(stochastic_string: str) -> float:
    events = stochastic_string.split(",")
    values = [float(event.split(":")[0]) for event in events]
    frequencies = [float(event.split(":")[1]) for event in events]
    return np.random.choice(values, p=frequencies)


def add_workload_args_to_results(results_df: pd.DataFrame):
    def get_row_parameters(row):
        task_path = row["tasks_path"]
        args = get_args_from_taskname(task_path)
        args["trial_id"] = row["trial_id"]
        return pd.Series(args)

    df = results_df.apply(get_row_parameters, axis=1)
    return results_df.merge(df, on="trial_id")


def get_name_from_args(arg_dict: dict, category="task") -> str:
    arg_string = ",".join([f"{key}={value}" for key, value in arg_dict.items()])
    task_path = f"{category}-{arg_string}"
    return task_path


def get_args_from_taskname(task_path: str) -> dict:
    arg_string = task_path.split("-")[1]
    arg_dict = {
        kv.split("=")[0]: float(kv.split("=")[1]) for kv in arg_string.split(",")
    }
    return arg_dict


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

    simulator_config = simulator_config.dump()
    omegaconf = OmegaConf.create(simulator_config["omegaconf"])

    n_allocated_tasks = 0
    tasks_scheduling_times = []
    allocated_tasks_scheduling_delays = []
    maximum_profit = 0
    realized_profit = 0

    log_tasks = []
    if omegaconf.logs.save:
        for task in tasks:
            task_dump = task.dump(budget_per_block=omegaconf.logs.verbose)

            result = error = None
            maximum_profit += task.profit
            if tasks_info.tasks_status[task.id] == ALLOCATED:
                n_allocated_tasks += 1
                realized_profit += task.profit
                tasks_scheduling_times.append(tasks_info.scheduling_time[task.id])
                allocated_tasks_scheduling_delays.append(
                    tasks_info.scheduling_delay.get(task.id, None)
                )

                result = tasks_info.result[task.id]
                error = tasks_info.error[task.id]

            task_dump.update(
                {
                    "allocated": tasks_info.tasks_status[task.id] == ALLOCATED,
                    "status": tasks_info.tasks_status[task.id],
                    "result": result,
                    "error": error,
                    "planning_time": tasks_info.planning_time[task.id],
                    "creation_time": tasks_info.creation_time[task.id],
                    "num_blocks": task.n_blocks,
                    "scheduling_time": tasks_info.scheduling_time.get(task.id, None),
                    "scheduling_delay": tasks_info.scheduling_delay.get(task.id, None),
                    "allocation_index": tasks_info.allocation_index.get(task.id, None),
                }
            )
            log_tasks.append(task_dump)

    log_blocks = []
    if omegaconf.logs.save:
        for block in blocks.values():
            log_blocks.append(block.dump())
    total_tasks = len(tasks)
    allocated_tasks_scheduling_delays = allocated_tasks_scheduling_delays

    datapoint = {
        "scheduler": omegaconf.scheduler.method,
        "solver": omegaconf.scheduler.solver,
        "scheduler_n": omegaconf.scheduler.n,
        "scheduler_metric": omegaconf.scheduler.metric,
        "T": omegaconf.scheduler.scheduling_wait_time,
        # "budget_utilization": bu,
        "data_lifetime": omegaconf.scheduler.data_lifetime,
        "block_selecting_policy": omegaconf.tasks.block_selection_policy,
        "n_allocated_tasks": n_allocated_tasks,
        "planner": omegaconf.scheduler.planner,
        "cache": omegaconf.scheduler.cache,
        "total_tasks": total_tasks,
        "realized_profit": realized_profit,
        "n_initial_blocks": omegaconf.blocks.initial_num,
        "maximum_profit": maximum_profit,
        "mean_task_per_block": omegaconf.tasks.avg_num_tasks_per_block,
        "path": omegaconf.tasks.path,
        "allocated_tasks_scheduling_delays": allocated_tasks_scheduling_delays,
        "initial_blocks": omegaconf.blocks.initial_num,
        "max_blocks": omegaconf.blocks.max_num,
        "tasks": log_tasks,
        "blocks": log_blocks,
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
    log_path = LOGS_PATH.joinpath(
        f"{datetime.now().strftime('%m%d-%H%M%S')}_{str(uuid.uuid4())[:6]}.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        raise NotImplementedError
    else:
        with open(log_path, "w") as fp:
            if compact:
                json_object = json.dumps(log_dict, separators=(",", ":"))
            else:
                json_object = json.dumps(log_dict, indent=4)

            fp.write(json_object)
