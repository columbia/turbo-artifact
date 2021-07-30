from privacypacking.budget.curves import *
from privacypacking.utils.utils import *


class Task:
    def __init__(self, id, num_blocks, type):
        self.id = id
        self.type = type
        self.budget_per_block = {}  # block_id -> Budget
        # Initialize all block demands to zero 
        for i in range(num_blocks):
            self.budget_per_block[i] = ZeroCurve().budget


def create_laplace_task(task_id, num_blocks, block_ids, noise):
    # Same curve for all blocks for now / same demands per block
    task = Task(task_id, num_blocks, LAPLACE)
    for block_id in block_ids:
        task.budget_per_block[block_id] = LaplaceCurve(laplace_noise=noise).budget
    return task


def create_gaussian_task(task_id, num_blocks, block_ids, sigma):
    # Same curve for all blocks for now / same demands per block
    task = Task(task_id, num_blocks, GAUSSIAN)
    for block_id in block_ids:
        task.budget_per_block[block_id] = GaussianCurve(sigma=sigma).budget
    return task


def create_subsamplegaussian_task(task_id, num_blocks, block_ids, ds, bs, epochs, s):
    # Same curve for all blocks for now / same demands per block
    task = Task(task_id, num_blocks, SUBSAMPLEGAUSSIAN)
    for block_id in block_ids:
        task.budget_per_block[block_id] = SubsampledGaussianCurve.from_training_parameters(ds, bs, epochs,
                                                                                           s).budget
    return task
