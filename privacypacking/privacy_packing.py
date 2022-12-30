import json
import os
import sys

import mlflow
import typer
import random
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import DEFAULT_CONFIG_FILE

from privacypacking.utils.utils import (
    LOGS_PATH,
    save_logs,
    save_mlflow_artifacts,
)

app = typer.Typer()


def privacypacking(omegaconf):
    default_omegaconf = OmegaConf.load(DEFAULT_CONFIG_FILE)
    custom_omegaconf = OmegaConf.create(omegaconf)
    omegaconf = OmegaConf.merge(default_omegaconf, custom_omegaconf)

    if omegaconf.enable_random_seed:
        random.seed(None)
        np.random.seed(None)
    else:
        random.seed(omegaconf.global_seed)
        np.random.seed(omegaconf.global_seed)

    if omegaconf.logs.mlflow:
        os.environ["MLFLOW_TRACKING_URI"] = str(LOGS_PATH.joinpath("mlruns"))
        mlflow.set_experiment(experiment_id="0")
        with mlflow.start_run():
            # You can also log nexted dicts individually if you prefer
            mlflow.log_params(OmegaConf.to_container(omegaconf))
            logs = Simulator(omegaconf).run()
            save_mlflow_artifacts(logs)
    else:
        logs = Simulator(omegaconf).run()

    save_logs(logs)
    return logs


@app.command()
def run(
    omegaconf: str = "privacypacking/config/debug_pmw_config.json",
    loguru_level: str = "INFO",
):

    # Try environment variable first, CLI arg otherwise
    level = os.environ.get("LOGURU_LEVEL", loguru_level)
    logger.remove()
    logger.add(sys.stdout, level=level)

    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    # Read config file
    with open(omegaconf) as f:
        omegaconf = json.load(f)
        privacypacking(omegaconf)


if __name__ == "__main__":
    app()
