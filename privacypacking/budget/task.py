from typing import Dict, Iterable

from privacypacking.budget.budget import Budget
from privacypacking.budget.curves import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
    ZeroCurve,
)
from privacypacking.utils.utils import *


class Task:
    def __init__(self, id: int, profit: float, budget_per_block: Dict[int, "Budget"]):
        self.id = id
        self.profit = profit
        self.budget_per_block = budget_per_block  # block_id -> Budget

    def get_budget(self, block_id: int) -> Budget:
        """
        Args:
            block_id (int): a block id

        Returns:
            Budget: the budget of the block if demanded by the task, else ZeroCurve
        """

        if block_id in self.budget_per_block:
            return self.budget_per_block[block_id]
        else:
            return ZeroCurve()


class UniformTask(Task):
    def __init__(
        self, id: int, profit: float, block_ids: Iterable[int], budget: Budget
    ):
        """
        A Task that requires (the same) `budget` for all blocks in `block_ids`
        """
        budget_per_block = {}
        for block_id in block_ids:
            budget_per_block[block_id] = budget
        super().__init__(id, profit, budget_per_block)


# Deprecated. Use UniformTask instead? (actually both are fine)
def create_laplace_task(task_id, num_blocks, block_ids, noise, profit=1):
    return UniformTask(
        id=task_id,
        profit=profit,
        block_ids=block_ids,
        budget=LaplaceCurve(noise),
    )


def create_gaussian_task(task_id, num_blocks, block_ids, sigma, profit=1):
    return UniformTask(
        id=task_id,
        profit=profit,
        block_ids=block_ids,
        budget=GaussianCurve(sigma),
    )


def create_subsamplegaussian_task(
    task_id, num_blocks, block_ids, ds, bs, epochs, s, profit=1
):
    return UniformTask(
        id=task_id,
        profit=profit,
        block_ids=block_ids,
        budget=SubsampledGaussianCurve.from_training_parameters(ds, bs, epochs, s),
    )
