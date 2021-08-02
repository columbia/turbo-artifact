from privacypacking.budget.curves import *
from privacypacking.utils.utils import *


# TODO: should we store a dict of blocks, to avoid `get_block_by_id`?
class Task:
    def __init__(self, id, num_blocks, block_ids, type):
        self.id = id
        self.type = type
        self.budget_per_block = {}  # block_id -> Budget
        self.num_blocks = (
            num_blocks  # current total number of blocks in the environment
        )
        self.block_ids = block_ids  # block ids requested by task
        # Initialize all block demands to zero
        for i in range(num_blocks):
            self.budget_per_block[i] = ZeroCurve().budget


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
