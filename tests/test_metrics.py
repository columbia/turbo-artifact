from os import setegid
from typing import List, Tuple

from loguru import logger

from privacypacking.budget import Block, Budget, Task
from privacypacking.budget.block_selection import RandomBlocks
from privacypacking.budget.curves import GaussianCurve, LaplaceCurve
from privacypacking.budget.task import UniformTask
from privacypacking.schedulers.budget_unlocking import UnlockingBlock
from privacypacking.schedulers.metrics import BatchOverflowRelevance


def setup_one_block() -> Tuple[List[Block], List[Task]]:
    b1 = UnlockingBlock(id=1, budget=Budget.from_epsilon_delta(10, 1e-5), n=1)
    t1 = UniformTask(
        id=1,
        profit=1,
        block_selection_policy=RandomBlocks,
        n_blocks=1,
        budget=LaplaceCurve(1.0),
    )
    t2 = UniformTask(
        id=2,
        profit=1,
        block_selection_policy=RandomBlocks,
        n_blocks=1,
        budget=GaussianCurve(1.0),
    )
    b1.unlock_budget()

    return [b1], [t1, t2] * 10


def test_batch_overflow():
    blocks, tasks = setup_one_block()
    logger.info(f"Testing batch overflow:{blocks}, {tasks}")

    # TODO: quick test
    assert True
