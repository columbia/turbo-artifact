import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

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

    df = pd.DataFrame(d).sort_values(
        ["block", "id", "allocated"], ascending=[True, True, False]
    )

    return df


def load_latest_scheduling_results() -> pd.DataFrame:
    exp_dirs = list(LOGS_PATH.glob("exp_*"))

    latest_exp_dir = max(exp_dirs, key=lambda x: x.name)

    return load_scheduling_dumps(latest_exp_dir.glob("**/*.json"))
