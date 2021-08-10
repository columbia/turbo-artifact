from typing import Dict, Iterable

from privacypacking.budget.budget import Budget
from privacypacking.budget.curves import (
    ZeroCurve,
)


class Task:
    def __init__(self, id: int, type: str, profit: float, budget_per_block: Dict[int, "Budget"]):
        self.id = id
        self.type = type
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
            self, id: int, type: str, profit: float, block_ids: Iterable[int], budget: Budget
    ):
        """
        A Task that requires (the same) `budget` for all blocks in `block_ids`
        """
        budget_per_block = {}
        for block_id in block_ids:
            budget_per_block[block_id] = budget
        super().__init__(id, type, profit, budget_per_block)
