import json
import os
import random
import time

import mlflow
import numpy as np
import simpy
import typer
from loguru import logger
from omegaconf import OmegaConf

from precycle.budget_accountant import BudgetAccountant, MockBudgetAccountant
from precycle.cache.cache import Cache, MockCache
from precycle.planner.min_cuts import MinCuts
from precycle.planner.no_cuts import NoCuts
from precycle.psql import PSQL, MockPSQL
from precycle.query_processor import QueryProcessor
from precycle.simulator import Blocks, ResourceManager, Tasks
from precycle.utils.utils import DEFAULT_CONFIG_FILE, LOGS_PATH, get_logs, save_logs, set_run_key

app = typer.Typer()


class Simulator:
    def __init__(self, omegaconf):
        self.env = simpy.Environment()

        # Initialize configuration
        default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
        omegaconf = OmegaConf.create(omegaconf)
        self.config = OmegaConf.merge(default_config, omegaconf)

        if self.config.logs.print_pid:
            # PID for profiling, sleep a bit to give time to attach the profiler
            print(f"PID: {os.getpid()}")
            time.sleep(3)

        if self.config.logs.mlflow:
            os.environ["MLFLOW_TRACKING_URI"] = str(LOGS_PATH.joinpath("mlruns"))
            try:
                mlflow.set_experiment(
                    experiment_name=self.config.logs.mlflow_experiment_id
                )
            except Exception:
                experiment_id = mlflow.create_experiment(name=self.config.logs.mlflow_experiment_id)
                print(f"New MLflow experiment created: {experiment_id}")

        try:
            with open(self.config.blocks.block_metadata_path) as f:
                blocks_metadata = json.load(f)
        except NameError:
            logger.error("Dataset metadata must have be created first..")
        assert blocks_metadata is not None
        self.config.update({"blocks_metadata": blocks_metadata})
        
        if self.config.mechanism.type == "TimestampsPMW":
            # Extend the attributes domain sizes with the domain size of the 'blocks' attribute
            max_blocks = int(self.config.blocks.max_num)
            # Must run only in the static case
            assert max_blocks == int(self.config.blocks.initial_num)
            pmw_attribute_names = self.config.blocks_metadata.attribute_names + ["blocks"]
            pmw_attributes_domain_sizes = self.config.blocks_metadata.attributes_domain_sizes + [max_blocks]
            pmw_domain_size = self.config.blocks_metadata.domain_size * max_blocks

        else:
            pmw_attribute_names = self.config.blocks_metadata.attribute_names
            pmw_attributes_domain_sizes = self.config.blocks_metadata.attributes_domain_sizes
            pmw_domain_size = self.config.blocks_metadata.domain_size

            
        self.config.blocks_metadata.update({"pmw_attribute_names": pmw_attribute_names,
                                            "pmw_attributes_domain_sizes": pmw_attributes_domain_sizes,
                                            "pmw_domain_size": pmw_domain_size})
        
        print(self.config.blocks_metadata.pmw_attribute_names)
        print(self.config.blocks_metadata.pmw_attributes_domain_sizes)
        print(self.config.blocks_metadata.pmw_domain_size)

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
            cache = MockCache(self.config)
        else:
            db = PSQL(self.config)
            budget_accountant = BudgetAccountant(self.config)
            cache = Cache(self.config)

        if self.config.planner.method == "MinCuts":
            planner = MinCuts(cache, budget_accountant, self.config)
        elif self.config.planner.method == "NoCuts":
            planner = NoCuts(cache, budget_accountant, self.config)

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
        config = OmegaConf.to_object(self.config)
        config["blocks_metadata"] = {}
        config["blocks"]["block_requests_pattern"] = {}

        key, _, _, _, _ = set_run_key(config)
        key += "_zip_" + str(config['tasks']['zipf_k'])
        with mlflow.start_run(run_name=key):
            # TODO: flatten dict to compare nested params
            mlflow.log_params(config)
            self.env.run()

            if self.config.logs.save:
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
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    omegaconf = OmegaConf.load(omegaconf)
    logs = Simulator(omegaconf).run()
    return logs


if __name__ == "__main__":
    app()
