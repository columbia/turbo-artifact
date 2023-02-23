import os
import sys
import typer

from loguru import logger
from omegaconf import OmegaConf

from precycle.server_tasks import TasksServer
from precycle.server_blocks import BlocksServer
from precycle.query_processor import QueryProcessor
from precycle.psql import PSQL, MockPSQL
from precycle.budget_accountant import BudgetAccountant, MockBudgetAccountant
from precycle.cache.laplace_cache import (
    DeterministicCache,
    MockDeterministicCache,
)

from precycle.utils.utils import DEFAULT_CONFIG_FILE


app = typer.Typer()


def precycle(custom_config):
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconfig = OmegaConf.create(custom_config)
    config = OmegaConf.merge(default_config, omegaconfig)
    print(config)

    if config.mock:
        db = MockPSQL(config)
        cache = MockDeterministicCache(config.cache)
        budget_accountant = MockBudgetAccountant(config=config.budget_accountant)
    else:
        db = PSQL(config)
        cache = DeterministicCache(config.cache)
        budget_accountant = BudgetAccountant(config=config.budget_accountant)

    # Initialize Query Processor
    query_processor = QueryProcessor(db, cache, budget_accountant, config)

    # TODO: make it  non blocking
    BlocksServer(db, budget_accountant, config.blocks_server).run()
    TasksServer(query_processor, budget_accountant, config.tasks_server).run()


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
