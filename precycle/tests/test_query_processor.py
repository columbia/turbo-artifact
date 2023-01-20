import typer
from omegaconf import OmegaConf
from precycle.task import Task
from precycle.query_processor import QueryProcessor
from precycle.sql_converter import SQLConverter
from precycle.cache.deterministic_cache import DeterministicCache
from precycle.budget_accounant import BudgetAccountant
from precycle.psql_connection import PSQLConnection
from precycle.utils.utils import DEFAULT_CONFIG_FILE

test = typer.Typer()

@test.command()
def test(
    omegaconf: str = "precycle/config/precycle.json",
):
    omegaconf = OmegaConf.load(omegaconf)
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconf = OmegaConf.create(omegaconf)
    config = OmegaConf.merge(default_config, omegaconf)

    query_vector = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 2],
        [0, 0, 0, 3],
        [0, 0, 0, 4],
        [0, 0, 0, 5],
        [0, 0, 0, 6],
        [0, 0, 0, 7],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 2],
        [0, 0, 1, 3],
        [0, 0, 1, 4],
        [0, 0, 1, 5],
        [0, 0, 1, 6],
        [0, 0, 1, 7],
        [0, 0, 2, 0],
        [0, 0, 2, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 3],
        [0, 0, 2, 4],
        [0, 0, 2, 5],
        [0, 0, 2, 6],
        [0, 0, 2, 7],
        [0, 0, 3, 0],
        [0, 0, 3, 1],
        [0, 0, 3, 2],
        [0, 0, 3, 3],
        [0, 0, 3, 4],
        [0, 0, 3, 5],
        [0, 0, 3, 6],
        [0, 0, 3, 7],
    ]

    num_requested_blocks = 1
    budget_accountant = BudgetAccountant(config=config.budget_accountant)
    num_blocks = budget_accountant.get_blocks_count()

    # Latest Blocks first
    requested_blocks = (num_blocks-num_requested_blocks, num_blocks-1)
    print(requested_blocks)

    task = Task(
        id=0,
        query_id=0,
        query_type="linear",
        query=query_vector,
        blocks=requested_blocks,
        n_blocks=num_requested_blocks,
        utility=100,
        utility_beta=0.0001,
        name=0,
    )
    sql_converter = SQLConverter(config.blocks.block_metadata_path)
    db = PSQLConnection(config, sql_converter)
    cache = DeterministicCache(config.cache)

    query_processor = QueryProcessor(
        db, cache, budget_accountant, sql_converter, config
    )
    run_metadata = query_processor.try_run_task(task)
    print(run_metadata)

if __name__ == "__main__":
    test()
