import simpy
import json
import typer
import random
import os
import sys
import numpy as np
from loguru import logger

from omegaconf import OmegaConf
from precycle.simulator import Blocks, ResourceManager, Tasks

from precycle.query_processor import QueryProcessor
from precycle.psql_connection import MockPSQLConnection, PSQLConnection
from precycle.budget_accounant import MockBudgetAccountant, BudgetAccountant

from precycle.cache.deterministic_cache import MockDeterministicCache, DeterministicCache
from precycle.cache.probabilistic_cache import MockProbabilisticCache

# from precycle.planner.ilp import ILP
from precycle.planner.max_cuts_planner import MaxCutsPlanner
from precycle.planner.min_cuts_planner import MinCutsPlanner

from precycle.utils.utils import DEFAULT_CONFIG_FILE


app = typer.Typer()


class Simulator:
    def __init__(self, config):
        self.env = simpy.Environment()

        # Initialize configuration
        omegaconf = OmegaConf.load(omegaconf)
        default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
        omegaconf = OmegaConf.create(omegaconf)
        config = OmegaConf.merge(default_config, omegaconf)

        try:
            with open(config.blocks.block_metadata_path) as f:
                blocks_metadata = json.load(f)
        except NameError:
            logger.error("Dataset metadata must have be created first..")
        assert blocks_metadata is not None
        config.update({"blocks_metadata": blocks_metadata})

        if config.enable_random_seed:
            random.seed(None)
            np.random.seed(None)
        else:
            random.seed(config.global_seed)
            np.random.seed(config.global_seed)


        # Initialize all components
        if config.mock:
            db = MockPSQLConnection(config)
            budget_accountant = MockBudgetAccountant(config)
            cache = globals()[f"Mock{config.cache.type}"](config)
        else:
            db = PSQLConnection(config)
            budget_accountant = BudgetAccountant(config)
            cache = globals()[config.cache.type](config)
        
        planner = globals()[config.planner.method](
            cache, budget_accountant, config
        )
        
        query_processor = QueryProcessor(db, cache, planner, budget_accountant, config)

        # Start the block and tasks consumers
        self.rm = ResourceManager(self.env, db, budget_accountant, query_processor, config)
        self.env.process(self.rm.start())

        # Start the block and tasks producers
        Blocks(self.env, self.rm)
        Tasks(self.env, self.rm)


    def run(self):
        self.env.run()

        # Rough estimate of the scheduler's performance
        # logs = get_logs(
        #     self.rm.scheduler.task_queue.tasks
        #     + list(self.rm.scheduler.tasks_info.allocated_tasks.values()),
        #     self.rm.scheduler.blocks,
        #     self.rm.scheduler.tasks_info,
        #     # list(self.rm.scheduler.tasks_info.allocated_tasks.keys()),
        #     self.omegaconf,
        #     scheduling_time=simulation_duration,
        #     scheduling_queue_info=self.rm.scheduler.scheduling_queue_info
        #     if hasattr(self.rm.scheduler, "scheduling_queue_info")
        #     else None,
        # )
        return


@app.command()
def run(
    omegaconf: str = "precycle/config/precycle.json",
    loguru_level: str = "INFO",
):

    # Try environment variable first, CLI arg otherwise
    level = os.environ.get("LOGURU_LEVEL", loguru_level)
    logger.remove()
    logger.add(sys.stdout, level=level)

    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    logs = Simulator(omegaconf).run()


if __name__ == "__main__":
    app()
