import os
import sys
import typer

from loguru import logger
from omegaconf import OmegaConf

from precycle.utils.utils import DEFAULT_CONFIG_FILE

from precycle.query_processor import QueryProcessor
from precycle.sql_converter import SQLConverter
from precycle.server_blocks import BlocksServer
from precycle.server_tasks import TasksServer
from precycle.budget_accounant import BudgetAccountant
from precycle.mock_budget_accounant import MockBudgetAccountant
from precycle.psql_connection import PSQLConnection
from precycle.mock_psql_connection import MockPSQLConnection
from precycle.cache.deterministic_cache import DeterministicCache
from precycle.cache.mock_deterministic_cache import MockDeterministicCache

# from precycle.cache.probabilistic_cache import ProbabilisticCache

app = typer.Typer()


def precycle(custom_config):
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconfig = OmegaConf.create(custom_config)
    config = OmegaConf.merge(default_config, omegaconfig)
    print(config)

    if config.mock:
        db = MockPSQLConnection(config)
        cache = MockDeterministicCache(config.cache)
        budget_accountant = MockBudgetAccountant(config=config.budget_accountant)
        sql_converter = None
    else:
        budget_accountant = BudgetAccountant(config=config.budget_accountant)
        sql_converter = SQLConverter(config.blocks.block_metadata_path)
        db = PSQLConnection(config, sql_converter)
        cache = DeterministicCache(config.cache)

    # Initialize Query Processor
    query_processor = QueryProcessor(
        db, cache, budget_accountant, sql_converter, config
    )
    
    # TODO: make it  non blocking
    BlocksServer(db, budget_accountant, config.blocks_server).run()  
    # TasksServer(query_processor, budget_accountant, config.tasks_server).run()



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

    omegaconf = OmegaConf.load(omegaconf)
    precycle(omegaconf)


if __name__ == "__main__":
    app()
