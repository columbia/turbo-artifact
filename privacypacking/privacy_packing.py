import json
import os
import sys

import mlflow
import typer
from loguru import logger
from omegaconf import OmegaConf

from privacypacking.config import Config
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import (
    DEFAULT_CONFIG_FILE,
    LOGS_PATH,
    save_logs,
    save_mlflow_artifacts,
)

app = typer.Typer()


def main(config):
    conf = Config(config)

    if conf.omegaconf.logs.mlflow:
        os.environ["MLFLOW_TRACKING_URI"] = str(LOGS_PATH.joinpath("mlruns"))
        with mlflow.start_run():
            # You can also log nexted dicts individually if you prefer
            mlflow.log_params(OmegaConf.to_container(conf.omegaconf))
            logs = Simulator(conf).run()
            save_mlflow_artifacts(logs)
            save_logs(conf, logs)
    else:
        logs = Simulator(conf).run()
        save_logs(conf, logs)


@app.command()
def run(
    config: str = "privacypacking/config/debug_pmw_config.json",
    loguru_level: str = "WARNING",
):
    # Try environment variable first, CLI arg otherwise
    level = os.environ.get("LOGURU_LEVEL", loguru_level)
    logger.remove(0)
    logger.add(sys.stderr, level=level)

    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    # Read config file
    with open(config) as f:
        config = json.load(f)
        main(config)


if __name__ == "__main__":
    app()
