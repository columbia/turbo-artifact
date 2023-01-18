import os
import sys
import typer
import psycopg2

from loguru import logger
from omegaconf import OmegaConf
from utils.utils import DEFAULT_CONFIG_FILE
from query_processor import QueryProcessor
from budget_accounant import BudgetAccountant

from server_blocks import BlocksServer
from server_tasks import TasksServer

app = typer.Typer()


def precycle(custom_config):
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconfig = OmegaConf.create(custom_config)
    config = OmegaConf.merge(default_config, omegaconfig)
    print(config)

    # Initialize the Budget Accountant
    # Keeps track of the privacy budgets of the blocks entering the database
    budget_accountant = BudgetAccountant(config=config.budget_accountant)

    # Initialize the PSQL connection
    try:
        # Connect to the PostgreSQL database server
        psql_conn = psycopg2.connect(
            host=config.postgres.host,
            database=config.postgres.database,
            user=config.postgres.username,
            password=config.postgres.password,
        )
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        exit(1)

    # Initialize Query Processor
    query_processor = QueryProcessor(psql_conn, budget_accountant, config)

    BlocksServer(psql_conn, budget_accountant, config).run()  # TODO: make it  non blocking
    #TasksServer(query_processor, budget_accountant, config).run()
    
    if psql_conn is not None:
        psql_conn.close()
        print("Database connection closed.")


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
