import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd
import yaml

from privacypacking.budget.budget import Budget
from privacypacking.utils.utils import LOGS_PATH


def load_ray_experiment(logs: Union[Path, str]) -> pd.DataFrame:
    results = []
    for run_result in logs.glob("**/result.json"):
        try:
            with open(run_result, "r") as f:
                d = json.load(f)
            results.append(d)
        except Exception:
            pass

    df = pd.DataFrame(results)

    return df


def load_latest_ray_experiment() -> pd.DataFrame:
    log_dirs = list(LOGS_PATH.joinpath("ray").iterdir())

    latest_log_dir = max(log_dirs, key=lambda x: x.name)

    # Noisy logs so we don't forget which directory we're using
    print(latest_log_dir)

    return load_ray_experiment(latest_log_dir)


def load_scheduling_dumps(
    json_log_paths: Iterable[Union[Path, str]], verbose=False
) -> pd.DataFrame:
    d = defaultdict(list)

    for p in json_log_paths:
        if verbose:
            print(p)
        try:
            with open(p) as f:
                run_dict = json.load(f)
            for t in run_dict["tasks"]:
                for block_id, block_budget in t["budget_per_block"].items():

                    d["id"].append(t["id"])
                    d["hashed_id"].append(hash(str(t["id"])) % 100)
                    d["allocated"].append(t["allocated"])
                    d["profit"].append(t["profit"])
                    d["realized_profit"].append(t["profit"] if t["allocated"] else 0)
                    d["scheduler"].append(
                        run_dict["config"]["scheduler_spec"]["method"]
                    )
                    d["total_blocks"].append(len(run_dict["blocks"]))
                    d["n_blocks"].append(len(t["budget_per_block"]))

                    d["block"].append(int(block_id))
                    d["epsilon"].append(block_budget["dp_budget"]["epsilon"])
                    d["block_selection"].append(
                        run_dict["config"]["tasks_spec"]["curve_distributions"][
                            "custom"
                        ]["read_block_selecting_policy_from_config"][
                            "block_selecting_policy"
                        ]
                    )
                    d["totalblocks_scheduler_selection"].append(
                        f"{d['total_blocks'][-1]}-{d['scheduler'][-1]}-{d['block_selection'][-1]}"
                    )
                    d["metric"].append(run_dict["config"]["scheduler_spec"]["metric"])
                    d["nblocks_maxeps"].append(
                        f"{d['n_blocks'][-1]}-{block_budget['orders']['64']:.3f}"
                    )
        except Exception as e:
            print(e)

    df = pd.DataFrame(d).sort_values(
        ["block", "id", "allocated"], ascending=[True, True, False]
    )

    return df


def load_scheduling_dumps_monoalpha(
    json_log_paths: Iterable[Union[Path, str]],
    verbose=False,
) -> pd.DataFrame:
    d = defaultdict(list)

    for p in json_log_paths:
        if verbose:
            print(p)
        with open(p) as f:
            run_dict = json.load(f)

        block_orders = run_dict["blocks"][0]["initial_budget"]["orders"]

        for t in run_dict["tasks"]:

            for block_id, block_budget in t["budget_per_block"].items():
                orders = t["budget_per_block"][block_id]["orders"]

                for alpha in block_orders:
                    # alpha = float(alpha) if alpha % 1 else int(alpha)
                    alpha = int(alpha)
                    d["alpha"].append(alpha)
                    d["blockid_alpha"].append(f"{int(block_id):03}-{alpha:02}")

                    # A fake alpha to separate blocks
                    if alpha == 0:
                        d["epsilon"].append(0)
                        d["normalized_epsilon"].append(0)
                    else:

                        d["epsilon"].append(orders[str(alpha)])
                        d["normalized_epsilon"].append(
                            orders[str(alpha)] / block_orders[str(alpha)]
                        )

                    d["id"].append(t["id"])
                    d["hashed_id"].append(hash(str(t["id"])) % 100)
                    d["allocated"].append(t["allocated"])
                    d["scheduler"].append(
                        run_dict["config"]["scheduler_spec"]["method"]
                    )
                    d["profit"].append(t["profit"])
                    d["realized_profit"].append(t["profit"] if t["allocated"] else 0)
                    d["total_blocks"].append(len(run_dict["blocks"]))
                    d["n_blocks"].append(len(t["budget_per_block"]))
                    d["creation_time"].append(t["creation_time"])
                    d["scheduling_time"].append(t["scheduling_time"])
                    d["allocation_index"].append(t["allocation_index"])
                    d["scheduling_delay"].append(t["scheduling_delay"])

                    d["block"].append(int(block_id))
                    d["block_selection"].append(
                        run_dict["config"]["tasks_spec"]["curve_distributions"][
                            "custom"
                        ]["read_block_selecting_policy_from_config"][
                            "block_selecting_policy"
                        ]
                    )
                    d["totalblocks_scheduler_selection"].append(
                        f"{d['total_blocks'][-1]}-{d['scheduler'][-1]}-{d['block_selection'][-1]}"
                    )
                    d["metric"].append(run_dict["config"]["scheduler_spec"]["metric"])
                    d["nblocks_maxeps"].append(
                        f"{d['n_blocks'][-1]}-{block_budget['orders']['64']:.3f}"
                    )
                    d["T"].append(
                        run_dict["config"]["scheduler_spec"]["scheduling_wait_time"]
                    ),
                    d["N"].append(run_dict["config"]["scheduler_spec"]["n"])
                    d["data_lifetime"].append(
                        run_dict["config"]["scheduler_spec"].get("data_lifetime", -1)
                    )

    df = pd.DataFrame(d).sort_values(
        ["blockid_alpha", "id", "allocated"], ascending=[True, True, False]
    )

    return df


def load_scheduling_dumps_alphas(
    json_log_paths: Iterable[Union[Path, str]],
    verbose=False,
) -> pd.DataFrame:
    d = defaultdict(list)

    for p in json_log_paths:
        if verbose:
            print(p)
        with open(p) as f:
            run_dict = json.load(f)

        block_orders = run_dict["blocks"][0]["initial_budget"]["orders"]

        for t in run_dict["tasks"]:
            for block_id, block_budget in t["budget_per_block"].items():
                orders = t["budget_per_block"][block_id]["orders"]

                # print(orders)
                # print(block_orders)

                # 1 / 0

                for alpha in [0, 4, 6, 8, 64]:
                    d["alpha"].append(alpha)
                    d["blockid_alpha"].append(f"{int(block_id):03}-{alpha:02}")

                    # A fake alpha to separate blocks
                    if alpha == 0:
                        d["epsilon"].append(0)
                        d["normalized_epsilon"].append(0)
                    else:

                        d["epsilon"].append(orders[str(alpha)])
                        d["normalized_epsilon"].append(
                            orders[str(alpha)] / block_orders[str(alpha)]
                        )

                    d["id"].append(t["id"])
                    d["hashed_id"].append(hash(str(t["id"])) % 100)
                    d["allocated"].append(t["allocated"])
                    d["scheduler"].append(
                        run_dict["config"]["scheduler_spec"]["method"]
                    )
                    d["profit"].append(t["profit"])
                    d["realized_profit"].append(t["profit"] if t["allocated"] else 0)
                    d["total_blocks"].append(len(run_dict["blocks"]))
                    d["n_blocks"].append(len(t["budget_per_block"]))
                    d["creation_time"].append(t["creation_time"])
                    d["scheduling_time"].append(t["scheduling_time"])
                    d["allocation_index"].append(t["allocation_index"])
                    d["scheduling_delay"].append(t["scheduling_delay"])

                    d["block"].append(int(block_id))
                    d["block_selection"].append(
                        run_dict["config"]["tasks_spec"]["curve_distributions"][
                            "custom"
                        ]["read_block_selecting_policy_from_config"][
                            "block_selecting_policy"
                        ]
                    )
                    d["totalblocks_scheduler_selection"].append(
                        f"{d['total_blocks'][-1]}-{d['scheduler'][-1]}-{d['block_selection'][-1]}"
                    )
                    d["metric"].append(run_dict["config"]["scheduler_spec"]["metric"])
                    d["nblocks_maxeps"].append(
                        f"{d['n_blocks'][-1]}-{block_budget['orders']['64']:.3f}"
                    )
                    d["T"].append(
                        run_dict["config"]["scheduler_spec"]["scheduling_wait_time"]
                    ),
                    d["N"].append(run_dict["config"]["scheduler_spec"]["n"])
                    d["data_lifetime"].append(
                        run_dict["config"]["scheduler_spec"].get("data_lifetime", -1)
                    )
    df = pd.DataFrame(d).sort_values(
        ["blockid_alpha", "id", "allocated"], ascending=[True, True, False]
    )

    return df


def load_scheduling_queue(expname="") -> pd.DataFrame:
    if not expname:
        exp_dirs = list(LOGS_PATH.glob("exp_*"))
        latest_exp_dir = max(exp_dirs, key=lambda x: x.name)
    else:
        latest_exp_dir = LOGS_PATH.joinpath(expname)
    d = defaultdict(list)

    for p in latest_exp_dir.glob("**/*.json"):
        print(p)
        try:
            with open(p) as f:
                run_dict = json.load(f)
                for step_info in run_dict["scheduling_queue_info"]:
                    d["scheduling_time"].append(step_info["scheduling_time"])
                    d["iteration_counter"].append(step_info["iteration_counter"])

                    # Store the raw lists for now
                    d["ids_and_metrics"].append(step_info["ids_and_metrics"])

                    # General config info
                    d["metric"].append(run_dict["config"]["scheduler_spec"]["metric"])
                    d["T"].append(
                        run_dict["config"]["scheduler_spec"]["scheduling_wait_time"]
                    ),
                    d["N"].append(run_dict["config"]["scheduler_spec"]["n"])
                    d["data_lifetime"].append(
                        run_dict["config"]["scheduler_spec"]["data_lifetime"]
                    )
        except Exception as e:
            print(e)

    df = pd.DataFrame(d).sort_values(
        ["scheduling_time", "iteration_counter"],
        ascending=[True, True],
    )

    return df


def load_tasks(expname="", validate=False, tasks_dir="") -> pd.DataFrame:
    if not expname:
        exp_dirs = list(LOGS_PATH.glob("exp_*"))
        latest_exp_dir = max(exp_dirs, key=lambda x: x.name)
    else:
        latest_exp_dir = LOGS_PATH.joinpath(expname)
    d = defaultdict(list)

    # TODO: other relevant info here?
    # TODO: task dir from PrivateKube's data path
    for p in latest_exp_dir.glob("**/*.json"):
        with open(p) as f:
            run_dict = json.load(f)
        for t in run_dict["tasks"]:
            block_budget = list(t["budget_per_block"].values())[0]
            d["id"].append(t["id"])
            d["first_block_id"] = min(
                [int(block_id) for block_id in t["budget_per_block"].keys()]
            )
            d["n_blocks"].append(len(t["budget_per_block"]))
            d["profit"].append(t["profit"])
            d["creation_time"].append(t["creation_time"])

            # NOTE: scheduler dependent
            # d["scheduling_time"].append(t["scheduling_time"])
            # d["scheduling_delay"].append(t["scheduling_delay"])
            # d["allocated"].append(t["allocated"])
            d["nblocks_maxeps"].append(
                f"{d['n_blocks'][-1]}-{block_budget['orders']['64']:.3f}"
            )
        if not validate:
            break
        else:
            raise NotImplementedError
    df = pd.DataFrame(d).sort_values("id")

    if tasks_dir:
        maxeps = {}
        for task_file in Path(tasks_dir).glob("*.yaml"):
            task_dict = yaml.safe_load(task_file.open("r"))
            maxeps[f"{task_dict['rdp_epsilons'][-1]:.3f}"] = task_file.stem
        maxeps

        def get_task_name(s):
            n, m = s.split("-")
            return f"{n}-{maxeps[m]}"

        df["task"] = df["nblocks_maxeps"].apply(get_task_name)

    return df


# TODO: load tasks too, add their names, join the dataframe. Double check that tasks are identical in all runs (optional).
# TODO: add the tasks on the simulation to show the order in the queue. Histogram/colors with slider?


def load_latest_scheduling_results(
    alphas=False,
    expname="",
    tasks_dir="",
    verbose=False,
) -> pd.DataFrame:

    if not expname:
        exp_dirs = list(LOGS_PATH.glob("exp_*"))
        latest_exp_dir = max(exp_dirs, key=lambda x: x.name)
    else:
        latest_exp_dir = LOGS_PATH.joinpath(expname)

    if not alphas:
        df = load_scheduling_dumps(latest_exp_dir.glob("**/result.json"), verbose)
    elif alphas == "monoalpha":
        df = load_scheduling_dumps_monoalpha(
            latest_exp_dir.glob("**/result.json"), verbose
        )
    else:
        df = load_scheduling_dumps_alphas(
            latest_exp_dir.glob("**/result.json"), verbose
        )

    if tasks_dir:
        maxeps = {}
        for task_file in Path(tasks_dir).glob("*.yaml"):
            task_dict = yaml.safe_load(task_file.open("r"))
            maxeps[f"{task_dict['rdp_epsilons'][-1]:.3f}"] = task_file.stem
        maxeps

        def get_task_name(s):
            n, m = s.split("-")
            return f"{n}-{maxeps[m]}"

        df["task"] = df["nblocks_maxeps"].apply(get_task_name)

    return df


def get_percentiles(delay_df, percentile_list):
    d = defaultdict(list)
    delay_df_na = delay_df.dropna()
    for percentile in percentile_list:
        for T in delay_df_na["T"].unique():
            for metric in delay_df_na["metric"].unique():
                delay_series = delay_df_na.query(f"metric == '{metric}' and T == {T}")[
                    "scheduling_delay"
                ]
                d["delay"].append(np.percentile(delay_series, percentile))
                d["percentile"].append(percentile)
                d["T"].append(T)
                d["scheduler"].append(metric)
    percentile_df = pd.DataFrame(d).sort_values(["T", "percentile"])
    return percentile_df
