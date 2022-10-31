import json
import os

import typer
from privacypacking.config import Config
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import DEFAULT_CONFIG_FILE, save_logs

app = typer.Typer()


def main(config):
    conf = Config(config)
    logs = Simulator(conf).run()
    save_logs(conf, logs)


@app.command()
def run(
    config: str = "privacypacking/config/debug_pmw_config.json",
    loguru_level: str = "WARNING",
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    # Read config file
    with open(config) as f:
        config = json.load(f)
        main(config)


if __name__ == "__main__":
    app()
