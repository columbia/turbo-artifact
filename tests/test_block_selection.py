from privacypacking.budget import Block
from privacypacking.budget.block_selection import (
    BlockSelectionException,
    ContiguousBlocksRandomOffset,
)


def test_contiguous_block_selection():
    policy = ContiguousBlocksRandomOffset
    n_blocks = 10
    n_blocks_task = 5
    blocks = [Block.from_epsilon_delta(i, 10.0, 1e-6) for i in range(n_blocks)]
    selected = policy.select_blocks(blocks=blocks, task_blocks_num=n_blocks_task)

    assert set(selected) <= set(range(n_blocks))

    assert len(selected) == n_blocks_task

    n_blocks = 2
    n_blocks_task = 5
    blocks = [Block.from_epsilon_delta(i, 10.0, 1e-6) for i in range(n_blocks)]

    got_exception = False
    try:
        selected = policy.select_blocks(blocks=blocks, task_blocks_num=n_blocks_task)
    except BlockSelectionException:
        got_exception = True

    assert got_exception
