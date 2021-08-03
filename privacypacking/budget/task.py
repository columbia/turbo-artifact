from typing import Iterable

from privacypacking.budget.budget import Budget
from privacypacking.budget.curves import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
    ZeroCurve,
)
from privacypacking.utils.utils import *


# TODO: should we store a dict of blocks, to avoid `get_block_by_id`?
class Task:
    def __init__(self, id, num_blocks, block_ids, type):
        self.id = id

        # TODO: make this optional? Some tasks are a combination, no type
        self.type = type
        self.budget_per_block = {}  # block_id -> Budget

        # TODO: I don't think that we need this field in general.
        # The process that generates the Task might need `num_blocks` to assign the blocks,
        # but the Task itself doesn't need to know anything else

        self.num_blocks = (
            num_blocks  # current total number of blocks in the environment
        )

        self.block_ids = block_ids  # block ids requested by task
        # Initialize all block demands to zero
        # TODO: I think that zero budget should be implicit, because we might want to
        # add more blocks over time. Also, it can be slow to keep track of all the blocks
        # even if the task only cares about 1 or 2 blocks.
        for i in range(num_blocks):
            self.budget_per_block[i] = ZeroCurve().budget

    def get_budget_or_zero(self, block_id: int) -> Budget:
        if block_id in self.budget_per_block:
            return self.budget_per_block[block_id]
        else:
            return ZeroCurve().budget


class UniformTask(Task):
    # For the general case
    def __init__(self, id: int, block_ids: Iterable[int], budget: Budget):
        self.id = id
        self.budget_per_block = {}
        for block_id in block_ids:
            self.budget_per_block[block_id] = budget.copy()


def create_laplace_task(task_id, num_blocks, block_ids, noise):
    # Same curve for all blocks for now / same demands per block
    task = Task(task_id, num_blocks, block_ids, LAPLACE)
    for block_id in block_ids:
        task.budget_per_block[block_id] = LaplaceCurve(laplace_noise=noise).budget
    return task


def create_gaussian_task(task_id, num_blocks, block_ids, sigma):
    # Same curve for all blocks for now / same demands per block
    task = Task(task_id, num_blocks, block_ids, GAUSSIAN)
    for block_id in block_ids:
        task.budget_per_block[block_id] = GaussianCurve(sigma=sigma).budget
    return task


def create_subsamplegaussian_task(task_id, num_blocks, block_ids, ds, bs, epochs, s):
    # Same curve for all blocks for now / same demands per block
    task = Task(task_id, num_blocks, block_ids, SUBSAMPLEGAUSSIAN)
    for block_id in block_ids:
        task.budget_per_block[
            block_id
        ] = SubsampledGaussianCurve.from_training_parameters(ds, bs, epochs, s).budget
    return task
