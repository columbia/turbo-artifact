from privacypacking.budget import Block
from privacypacking.budget.block_selection import (
    BlockSelectionException,
    ContiguousBlocksRandomOffset,
    Zeta,
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


def test_zeta():

    n_blocks = 10
    n_blocks_task = 5
    blocks = [Block.from_epsilon_delta(i, 10.0, 1e-6) for i in range(n_blocks)]

    for s in [0.5, 1.0, 2.0, 10]:
        for run in range(10):
            policy = Zeta(s=s)
            selected = policy.select_blocks(
                blocks=blocks, task_blocks_num=n_blocks_task
            )

            print(selected)
        print()


if __name__ == "__main__":
    # test_contiguous_block_selection()
    test_zeta()
