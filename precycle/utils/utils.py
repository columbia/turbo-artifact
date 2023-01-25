import json
import uuid
import mlflow
from pathlib import Path
from datetime import datetime
import pandas as pd
from precycle.utils.plot import plot_budget_utilization_per_block, plot_task_status
from precycle.budget.renyi_budget import RenyiBudget

CUSTOM_LOG_PREFIX = "custom_log_prefix"
REPO_ROOT = Path(__file__).parent.parent.parent
LOGS_PATH = REPO_ROOT.joinpath("logs")
RAY_LOGS = LOGS_PATH.joinpath("ray")
DEFAULT_CONFIG_FILE = REPO_ROOT.joinpath("precycle/config/default.yaml")

FAILED = "failed"
PENDING = "pending"
FINISHED = "finished"


def mlflow_log(key, value, step):
    mlflow_run = mlflow.active_run()
    if mlflow_run:
        mlflow.log_metric(
            key,
            value,
            step=step,
        )


def get_blocks_size(blocks, blocks_metadata):
    if isinstance(blocks, tuple):
        num_blocks = blocks[1] - blocks[0] + 1
        if "block_size" in blocks_metadata:
            # All blocks have the same size
            n = num_blocks * blocks_metadata["block_size"]
        else:
            n = sum(
                [
                    float(blocks_metadata["blocks"][str(id)]["size"])
                    for id in range(blocks[0], blocks[1] + 1)
                ]
            )
        return n
    else:
        return float(blocks_metadata["blocks"][str(blocks)]["size"])


def load_logs(log_path: str, relative_path=True) -> dict:
    full_path = Path(log_path)
    if relative_path:
        full_path = LOGS_PATH.joinpath(log_path)
    with open(full_path, "r") as f:
        logs = json.load(f)
    return logs


def get_logs(
    tasks_info,
    block_budgets_info,
    config_dict,
    **kwargs,
) -> dict:

    n_allocated_tasks = 0
    hard_queries = 0
    for task_info in tasks_info:
        if task_info["status"] == FINISHED:
            n_allocated_tasks += 1
            if task_info["run_metadata"]["hard_query"]:
                hard_queries += 1

    blocks_initial_budget = RenyiBudget.from_epsilon_delta(
        epsilon=config_dict["budget_accountant"]["epsilon"],
        delta=config_dict["budget_accountant"]["delta"],
        alpha_list=config_dict["budget_accountant"]["alphas"],
    ).dump()

    workload = pd.read_csv(config_dict["tasks"]["path"], header=None)
    query_pool_size = len(workload) - 1
    config = {}
    config.update(
        {
            "n_allocated_tasks": n_allocated_tasks,
            "hard_queries": hard_queries,
            "total_tasks": len(tasks_info),
            "cache": config_dict["cache"]["type"],
            "planner": config_dict["planner"]["method"],
            "workload_path": config_dict["tasks"]["path"],
            "query_pool_size": query_pool_size,
            "tasks_info": tasks_info,
            "block_budgets_info": block_budgets_info,
            "blocks_initial_budget": blocks_initial_budget,
            "config": config_dict,
        }
    )

    # Any other thing to log
    for key, value in kwargs.items():
        config[key] = value
    return config


def save_logs(log_dict):
    log_path = LOGS_PATH.joinpath(
        f"{datetime.now().strftime('%m%d-%H%M%S')}_{str(uuid.uuid4())[:6]}.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fp:
        json_object = json.dumps(log_dict, indent=4)
        fp.write(json_object)


def save_mlflow_artifacts(log_dict):
    """
    Write down some figures directly in Mlflow instead of having to fire Plotly by hand in a notebook
    See also: `analysis.py`
    """
    # TODO: save in a custom dir when we run with Ray?
    artifacts_dir = LOGS_PATH.joinpath("mlflow_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plot_budget_utilization_per_block(block_log=log_dict["blocks"]).write_html(
        artifacts_dir.joinpath("budget_utilization.html")
    )
    plot_task_status(task_log=log_dict["tasks"]).write_html(
        artifacts_dir.joinpath("task_status.html")
    )

    mlflow.log_artifacts(artifacts_dir)
