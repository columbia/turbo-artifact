import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

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
    return pd.DataFrame(results)


def load_latest_ray_experiment() -> pd.DataFrame:
    log_dirs = list(LOGS_PATH.joinpath("ray").iterdir())

    latest_log_dir = max(log_dirs, key=lambda x: x.name)

    # Noisy logs so we don't forget which directory we're using
    print(latest_log_dir)

    return load_ray_experiment(latest_log_dir)


def load_scheduling_dumps(json_log_paths: Iterable[Union[Path, str]]) -> pd.DataFrame:
    d = defaultdict(list)

    for p in json_log_paths:
        print(p)
        with open(p) as f:
            run_dict = json.load(f)
        for t in run_dict["tasks"]:
            for block_id, block_budget in t["budget_per_block"].items():

                d["id"].append(t["id"])
                d["hashed_id"].append(hash(str(t["id"])) % 100)
                d["allocated"].append(t["allocated"])
                d["scheduler"].append(
                    run_dict["simulator_config"]["scheduler_spec"]["method"]
                )
                d["total_blocks"].append(len(run_dict["blocks"]))
                d["n_blocks"].append(len(t["budget_per_block"]))

                d["block"].append(int(block_id))
                d["epsilon"].append(block_budget["dp_budget"]["epsilon"])
                d["block_selection"].append(
                    run_dict["simulator_config"]["tasks_spec"]["curve_distributions"][
                        "custom"
                    ]["read_block_selecting_policy_from_config"][
                        "block_selecting_policy"
                    ]
                )
                d["totalblocks_scheduler_selection"].append(
                    f"{d['total_blocks'][-1]}-{d['scheduler'][-1]}-{d['block_selection'][-1]}"
                )
                d["metric"].append(
                    run_dict["simulator_config"]["scheduler_spec"]["metric"]
                )
                d["nblocks_maxeps"].append(
                    f"{d['n_blocks'][-1]}-{block_budget['orders']['64']:.3f}"
                )

    df = pd.DataFrame(d).sort_values(
        ["block", "id", "allocated"], ascending=[True, True, False]
    )

    return df


def load_scheduling_dumps_alphas(
    json_log_paths: Iterable[Union[Path, str]]
) -> pd.DataFrame:
    d = defaultdict(list)

    for p in json_log_paths:
        print(p)
        with open(p) as f:
            run_dict = json.load(f)

        block_orders = run_dict["blocks"][0]["initial_budget"]["orders"]

        for t in run_dict["tasks"]:
            for block_id, block_budget in t["budget_per_block"].items():
                orders = t["budget_per_block"][block_id]["orders"]

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
                        run_dict["simulator_config"]["scheduler_spec"]["method"]
                    )
                    d["total_blocks"].append(len(run_dict["blocks"]))
                    d["n_blocks"].append(len(t["budget_per_block"]))
                    d["creation_time"].append(t["creation_time"])
                    d["scheduling_time"].append(t["scheduling_time"])
                    d["scheduling_delay"].append(t["scheduling_delay"])

                    d["block"].append(int(block_id))
                    d["block_selection"].append(
                        run_dict["simulator_config"]["tasks_spec"][
                            "curve_distributions"
                        ]["custom"]["read_block_selecting_policy_from_config"][
                            "block_selecting_policy"
                        ]
                    )
                    d["totalblocks_scheduler_selection"].append(
                        f"{d['total_blocks'][-1]}-{d['scheduler'][-1]}-{d['block_selection'][-1]}"
                    )
                    d["metric"].append(
                        run_dict["simulator_config"]["scheduler_spec"]["metric"]
                    )
                    d["nblocks_maxeps"].append(
                        f"{d['n_blocks'][-1]}-{block_budget['orders']['64']:.3f}"
                    )
                    d["T"].append(
                        run_dict["simulator_config"]["scheduler_spec"][
                            "scheduling_wait_time"
                        ]
                    ),
                    d["N"].append(run_dict["simulator_config"]["scheduler_spec"]["n"])
                    d["data_lifetime"].append(
                        run_dict["simulator_config"]["scheduler_spec"]["data_lifetime"]
                    )

    df = pd.DataFrame(d).sort_values(
        ["blockid_alpha", "id", "allocated"], ascending=[True, True, False]
    )

    return df


def load_latest_scheduling_results(alphas=False, expname="") -> pd.DataFrame:

    if not expname:
        exp_dirs = list(LOGS_PATH.glob("exp_*"))
        latest_exp_dir = max(exp_dirs, key=lambda x: x.name)
    else:
        latest_exp_dir = LOGS_PATH.joinpath(expname)

    if not alphas:
        return load_scheduling_dumps(latest_exp_dir.glob("**/*.json"))
    return load_scheduling_dumps_alphas(latest_exp_dir.glob("**/*.json"))
