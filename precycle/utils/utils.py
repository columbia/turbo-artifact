import json
import uuid
import math
import mlflow
from pathlib import Path
from datetime import datetime
import pandas as pd

# from precycle.utils.plot import plot_budget_utilization_per_block, plot_task_status
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


def satisfies_constraint(blocks, branching_factor=2):
    """
    Checks if <blocks> satisfies the binary structure constraint
    """
    n = blocks[1] - blocks[0] + 1
    if not math.log(n, branching_factor).is_integer():
        return False
    if (blocks[0] % n) != 0:
        return False
    return True


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
    total_histogram_runs = 0
    total_laplace_runs = 0
    total_sv_misses = 0
    total_sv_checks = 0

    # Finally logging only a small number of tasks for faster analysis
    tasks_to_log = []
    accummulated_budget_per_block = {}

    for task_info in tasks_info:

        if task_info["status"] == FINISHED:
            n_allocated_tasks += 1

            run_metadata = task_info["run_metadata"]
            histogram_runs = laplace_runs = 0
            run_types_list = run_metadata["run_types"]
            for run_types in run_types_list:
                for run_type in run_types.values():
                    if run_type == "Laplace":
                        laplace_runs += 1
                    elif run_type == "Histogram":
                        histogram_runs += 1

            total_laplace_runs += laplace_runs
            total_histogram_runs += histogram_runs

            # TODO: clean this part - it's terrible
            sv_check_status_list = run_metadata["sv_check_status"]
            for sv_check_status in sv_check_status_list:
                if sv_check_status == False:
                    total_sv_misses += 1
                total_sv_checks += 1

            total_budget_per_block = {}
            # total budget per block keeps the budget spent for a task across all of its trials
            budget_per_block_list = run_metadata["budget_per_block"]
            for budget_per_block in budget_per_block_list:
                for block, budget in budget_per_block.items():
                    if block not in total_budget_per_block:
                        total_budget_per_block[block] = budget
                    else:
                        total_budget_per_block[block] += budget

            # Global budget per block holds the accumulated budget per block up to this point
            for block, budget in total_budget_per_block.items():
                if block not in accummulated_budget_per_block:
                    accummulated_budget_per_block[block] = budget
                else:
                    accummulated_budget_per_block[block] += budget

            accummulated_budget_per_block_dump = {}
            for block in accummulated_budget_per_block:
                accummulated_budget_per_block_dump[block] = accummulated_budget_per_block[
                    block
                ].dump()

            for block in total_budget_per_block:
                total_budget_per_block[block] = total_budget_per_block[block].dump()

            task_info["run_metadata"]["budget_per_block"] = total_budget_per_block
            task_info["run_metadata"].update(
                {"accummulated_budget_per_block": accummulated_budget_per_block_dump}
            )
            task_info.update(
                {
                    "laplace_runs": laplace_runs,
                    "histogram_runs": histogram_runs,
                }
            )

            # if task_info["id"] % int(config_dict["logs"]["log_every_n_tasks"]) == 0:
            tasks_to_log.append(task_info)

    blocks_initial_budget = RenyiBudget.from_epsilon_delta(
        epsilon=config_dict["budget_accountant"]["epsilon"],
        delta=config_dict["budget_accountant"]["delta"],
        alpha_list=config_dict["budget_accountant"]["alphas"],
    ).dump()

    workload = pd.read_csv(config_dict["tasks"]["path"])
    query_pool_size = len(workload["query_id"].unique())
    config = {}

    if config_dict["cache"]["type"] == "DeterministicCache":
        cache_type = "DeterministicCache"
        heuristic = ""
    elif config_dict["cache"]["type"] == "ProbabilisticCache":
        cache_type = "ProbabilisticCache"
        heuristic = ""
    else:
        cache_type = "MixedCache"
        heuristic = config_dict["cache"]["probabilistic_cfg"]["heuristic"]

    config.update(
        {
            "n_allocated_tasks": n_allocated_tasks,
            "total_tasks": len(tasks_info),
            "total_sv_misses": total_sv_misses,
            "total_sv_checks": total_sv_checks,
            "total_histogram_runs": total_histogram_runs,
            "total_laplace_runs": total_laplace_runs,
            "cache": cache_type,
            "planner": config_dict["planner"]["method"],
            "workload_path": config_dict["tasks"]["path"],
            "query_pool_size": query_pool_size,
            "tasks_info": tasks_to_log,
            "block_budgets_info": block_budgets_info,
            "blocks_initial_budget": blocks_initial_budget,
            "zipf_k": config_dict["tasks"]["zipf_k"],
            "heuristic": heuristic,
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
