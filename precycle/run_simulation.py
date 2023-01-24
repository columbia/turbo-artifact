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

from precycle.cache.deterministic_cache import (
    MockDeterministicCache,
    DeterministicCache,
)
from precycle.cache.probabilistic_cache import (
    MockProbabilisticCache,
    ProbabilisticCache,
)

from precycle.query_converter import TensorConverter, SQLConverter

# from precycle.planner.ilp import ILP
from precycle.planner.max_cuts_planner import MaxCutsPlanner
from precycle.planner.min_cuts_planner import MinCutsPlanner

from precycle.utils.utils import (
    DEFAULT_CONFIG_FILE,
    LOGS_PATH,
    save_logs,
    save_mlflow_artifacts,
)


app = typer.Typer()


class Simulator:
    def __init__(self, omegaconf):
        self.env = simpy.Environment()

        # Initialize configuration
        omegaconf = OmegaConf.load(omegaconf)
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
            cache = globals()[f"Mock{self.config.cache.type}"](self.config)
            query_converter = TensorConverter(blocks_metadata)
        else:
            db = PSQL(self.config)
            budget_accountant = BudgetAccountant(self.config)
            cache = globals()[self.config.cache.type](self.config)
            query_converter = SQLConverter(blocks_metadata)

        planner = globals()[self.config.planner.method](
            cache, budget_accountant, self.config
        )

        query_processor = QueryProcessor(
            db, cache, planner, budget_accountant, query_converter, self.config
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
        mlflow.start_run()

        self.env.run()

        config = OmegaConf.to_object(self.config)
        config["blocks_metadata"] = {}

        # Collecting logs
        logs = {}
        logs["tasks_info"] = self.rm.query_processor.tasks_info
        logs["block_budgets_info"] = self.rm.budget_accountant.dump()
        logs["config"] = config

        mlflow.log_params(config)
        mlflow.end_run()
        return logs


@app.command()
def run(
    omegaconf: str = "precycle/config/precycle.json",
    loguru_level: str = "INFO",
):
    # Try environment variable first, CLI arg otherwise
    # level = os.environ.get("LOGURU_LEVEL", loguru_level)
    # logger.remove()
    # logger.add(sys.stdout, level=loguru_level)

    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    logs = Simulator(omegaconf).run()
    save_logs(logs)


if __name__ == "__main__":
    app()
