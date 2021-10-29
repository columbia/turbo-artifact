import json
from pathlib import Path
from typing import Union

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
