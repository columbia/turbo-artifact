import os
import sys
import json
import typer
import simpy
import random
import numpy as np
import mlflow
from loguru import logger
from omegaconf import OmegaConf

from precycle.simulator import Blocks, ResourceManager, Tasks

from precycle.query_processor import QueryProcessor
from precycle.psql import MockPSQL, PSQL
from precycle.budget_accountant import MockBudgetAccountant, BudgetAccountant

from precycle.cache.combined_cache import MockCombinedCache

# from precycle.planner.ilp import ILP
from precycle.planner.max_cuts_planner import MaxCutsPlanner
from precycle.planner.min_cuts_planner import MinCutsPlanner

from precycle.utils.utils import (
    get_logs,
    save_logs,
    LOGS_PATH,
    DEFAULT_CONFIG_FILE,
    save_mlflow_artifacts,
)


app = typer.Typer()


class Simulator:
    def __init__(self, omegaconf):
        self.env = simpy.Environment()

        # Initialize configuration
        default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
        omegaconf = OmegaConf.create(omegaconf)
        self.config = OmegaConf.merge(default_config, omegaconf)

        if self.config.logs.mlflow:
            os.environ["MLFLOW_TRACKING_URI"] = str(LOGS_PATH.joinpath("mlruns"))
            mlflow.set_experiment(experiment_id="768944864734992566")

        try:
            with open(self.config.blocks.block_metadata_path) as f:
                blocks_metadata = json.load(f)
        except NameError:
            logger.error("Dataset metadata must have be created first..")
        assert blocks_metadata is not None
        self.config.update({"blocks_metadata": blocks_metadata})

        if self.config.enable_random_seed:
            random.seed(None)
            np.random.seed(None)
        else:
            random.seed(self.config.global_seed)
            np.random.seed(self.config.global_seed)

        # Initialize all components
        if self.config.mock:
            db = MockPSQL(self.config)
            budget_accountant = MockBudgetAccountant(self.config)
            cache = MockCombinedCache(self.config)
        # else:
        #     db = PSQL(self.config)
        #     budget_accountant = BudgetAccountant(self.config)
        #     cache = globals()[self.config.cache.type](self.config)

        planner = globals()[self.config.planner.method](
            cache, budget_accountant, self.config
        )

        query_processor = QueryProcessor(
            db, cache, planner, budget_accountant, self.config
        )

        # Start the block and tasks consumers
        self.rm = ResourceManager(
            self.env, db, budget_accountant, query_processor, self.config
        )
        self.env.process(self.rm.start())

        # Start the block and tasks producers
        Blocks(self.env, self.rm)
        Tasks(self.env, self.rm)

    def run(self):

        logs = None

        with mlflow.start_run():
            self.env.run()

            if self.config.logs.save:
                config = OmegaConf.to_object(self.config)
                config["blocks_metadata"] = {}
                mlflow.log_params(config)
                logs = get_logs(
                    self.rm.query_processor.tasks_info,
                    self.rm.budget_accountant.dump(),
                    config,
                )
                save_logs(logs)

        return logs


@app.command()
def run_simulation(
    omegaconf: str = "precycle/config/precycle.json",
    loguru_level: str = "INFO",
):
    # Try environment variable first, CLI arg otherwise
    # level = os.environ.get("LOGURU_LEVEL", loguru_level)
    # logger.remove()
    # logger.add(sys.stdout, level=loguru_level)

    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    omegaconf = OmegaConf.load(omegaconf)
    logs = Simulator(omegaconf).run()
    return logs


if __name__ == "__main__":
    app()
