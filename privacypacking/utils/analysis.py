import json
from pathlib import Path
from typing import Union

import pandas as pd


def load_ray_experiment(logs: Union[Path, str]) -> pd.DataFrame:
    results = []
    for run_result in logs.glob("**/result.json"):
        with open(run_result, "r") as f:
            d = json.load(f)
        results.append(d)
    return pd.DataFrame(results)