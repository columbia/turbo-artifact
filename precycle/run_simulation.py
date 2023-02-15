import json
import math
import os
import random
import sys

import mlflow
import numpy as np
import simpy
import typer
from loguru import logger
from omegaconf import OmegaConf

from precycle.budget_accountant import BudgetAccountant, MockBudgetAccountant
from precycle.cache.combined_cache import MockCombinedCache
from precycle.planner.ilp import ILP
from precycle.planner.min_cuts import MinCuts
from precycle.psql import PSQL, MockPSQL
from precycle.query_processor import QueryProcessor
from precycle.simulator import Blocks, ResourceManager, Tasks
from precycle.utils.compute_utility_curve import probabilistic_compute_utility_curve
from precycle.utils.utils import (
    DEFAULT_CONFIG_FILE,
    LOGS_PATH,
    get_logs,
    save_logs,
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
            try:
                mlflow.set_experiment(experiment_id="768944864734992566")
            except Exception:
                try:
                    mlflow.set_experiment(
                        experiment_id=self.config.logs.mlflow_experiment_id
                    )
                except Exception:
                    experiment_id = mlflow.create_experiment(name="precycle")
                    print(f"New MLflow experiment created: {experiment_id}")

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

        if not self.config.cache.type == "DeterministicCache":
            # This is the global accuracy supported by the probabilistic cache.
            # If a query comes requesting more accuracy than that it won't be able to serve it.
            assert self.config.cache.probabilistic_cfg.max_pmw_k is not None
            assert self.config.cache.probabilistic_cfg.alpha is not None
            assert self.config.cache.probabilistic_cfg.beta is not None

            if (
                self.config.cache.type == "CombinedCache"
                and self.config.blocks.max_num > 1
            ):
                # Mixing Up Deterministic With Probabilistic runs - union bound over the two
                # We need to change the beta of the probabilistic cache to: b = 1 - math.sqrt(1 - b)
                b = self.config.cache.probabilistic_cfg.beta
                self.config.cache.probabilistic_cfg.beta = 1 - math.sqrt(1 - b)

            self.config.cache.update(
                {
                    "pmw_accuracy": {
                        "alpha": self.config.cache.probabilistic_cfg.alpha,
                        "beta": self.config.cache.probabilistic_cfg.beta,
                        "max_pmw_k": self.config.cache.probabilistic_cfg.max_pmw_k,
                    }
                }
            )

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
