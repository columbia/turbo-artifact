import json
import typer
from loguru import logger
from precycle.task import Task
from omegaconf import OmegaConf

from precycle.query_processor import QueryProcessor
from precycle.psql import MockPSQL
from precycle.budget_accountant import MockBudgetAccountant

from precycle.planner.ilp import ILP
from precycle.cache.combined_cache import MockCombinedCache

from precycle.utils.compute_utility_curve import probabilistic_compute_utility_curve

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

    try:
        with open(config.blocks.block_metadata_path) as f:
            blocks_metadata = json.load(f)
    except NameError:
        logger.error("Dataset metadata must have be created first..")
    assert blocks_metadata is not None
    config.update({"blocks_metadata": blocks_metadata})

    if not config.cache.type == "DeterministicCache":
        # This is the global accuracy supported by the probabilistic cache.
        # If a query comes requesting more accuracy than that it won't be able to serve it.
        assert config.cache.probabilistic_cfg.max_pmw_k is not None
        assert config.cache.probabilistic_cfg.alpha is not None
        assert config.cache.probabilistic_cfg.beta is not None

        # No aggregations in min_cuts, no need for more powerful PMWs than needed
        if config.planner.method == "min_cuts":
            config.cache.probabilistic_cfg.max_pmw_k = 1

        pmw_alpha, pmw_beta = probabilistic_compute_utility_curve(
            config.cache.probabilistic_cfg.alpha,
            config.cache.probabilistic_cfg.beta,
            config.cache.probabilistic_cfg.max_pmw_k,
        )
        config.cache.update({"pmw_accuracy": {"alpha": pmw_alpha, "beta": pmw_beta}})
        print(config.cache)

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

    db = MockPSQL(config)
    budget_accountant = MockBudgetAccountant(config)
    cache = MockCombinedCache(config)
    planner = ILP(cache, budget_accountant, config)
    query_processor = QueryProcessor(db, cache, planner, budget_accountant, config)

    # Insert two blocks
    block_data_path = config.blocks.block_data_path + "/block_0.csv"
    db.add_new_block(block_data_path)
    budget_accountant.add_new_block_budget()
    # block_data_path = config.blocks.block_data_path + "/block_1.csv"
    # db.add_new_block(block_data_path)
    # budget_accountant.add_new_block_budget()

    # Initialize Task
    num_requested_blocks = 1
    num_blocks = budget_accountant.get_blocks_count()
    assert num_blocks > 0

    # Latest Blocks first
    requested_blocks = (num_blocks - num_requested_blocks, num_blocks - 1)
    # print(requested_blocks)

    utility = 0.05
    utility_beta = 0.0001

    task = Task(
        id=0,
        query_id=0,
        query_type="linear",
        query=query_vector,
        blocks=requested_blocks,
        n_blocks=num_requested_blocks,
        utility=utility,
        utility_beta=utility_beta,
        name=0,
    )

    run_metadata = query_processor.try_run_task(task)
    print(run_metadata)


if __name__ == "__main__":
    test()
