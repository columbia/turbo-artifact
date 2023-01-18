import os
import sys
import typer
import psycopg2

from loguru import logger
from omegaconf import OmegaConf
from precycle.utils.utils import DEFAULT_CONFIG_FILE
from precycle.query_processor import QueryProcessor
from precycle.budget_accounant import BudgetAccountant
from precycle.sql_converter import SQLConverter
from precycle.server_blocks import server_blocks
from precycle.server_tasks import server_tasks

app = typer.Typer()


def pricycle(custom_config):
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    custom_config = OmegaConf.create(custom_config)
    config = OmegaConf.merge(default_config, custom_config)

    # Initialize the Budget Accountant
    # Keeps track of the privacy budgets of the blocks entering the database
    budget_accountant = BudgetAccountant(config=config.budget_accountant)

    # Initialize the PSQL connection
    try:
        # Connect to the PostgreSQL database server
        psql_conn = psycopg2.connect(
            host=config.postgres.host,
            database=config.postgres.database,
            user=config.postgres.user,
            password=config.postgres.password,
        )
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        exit(1)


    sql_converter = SQLConverter(config.blocks_server.block_metadata_path)

    # Initialize Query Processor
    query_processor = QueryProcessor(psql_conn, budget_accountant, sql_converter, config)

    # TODO:make it  non blocking
    server_blocks(psql_conn, budget_accountant, config.blocks_server)
    server_tasks(query_processor, budget_accountant, config.tasks_server)
    
    if psql_conn is not None:
        psql_conn.close()
        print("Database connection closed.")


@app.command()
def run(
    omegaconf: str = "precycle/config/pricycle.json",
    loguru_level: str = "INFO",
):

    # Try environment variable first, CLI arg otherwise
    level = os.environ.get("LOGURU_LEVEL", loguru_level)
    logger.remove()
    logger.add(sys.stdout, level=level)

    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    pricycle(omegaconf)


if __name__ == "__main__":
    app()
