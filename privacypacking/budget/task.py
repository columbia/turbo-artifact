from typing import Iterable, Any

from privacypacking.budget.budget import Budget
from privacypacking.budget.curves import (
    ZeroCurve,
)
from privacypacking.budget.block_selection import BlockSelectionPolicy


class Task:
    def __init__(
        self,
        id: int,
        profit: float,
        block_selection_policy: BlockSelectionPolicy,
        n_blocks: int,
    ):
        self.id = id
        self.profit = profit
        self.block_selection_policy = block_selection_policy
        self.n_blocks = n_blocks
        # Scheduler dynamically updates the variables below
        self.budget_per_block = {}
        self.cost = 0

    def get_efficiency(self, cost):
        efficiency = 0
        try:
            efficiency = self.profit / cost
        except ZeroDivisionError as err:
            print("Handling run-time error:", err)
        return efficiency

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

    def set_budget_per_block(self, block_ids: Iterable[int]):
        pass

    def dump(self):
        return {
            "id": self.id,
            "profit": self.profit,
            "budget_per_block": {
                block_id: budget.dump()
                for block_id, budget in self.budget_per_block.items()
            },
        }


class UniformTask(Task):
    def __init__(
        self,
        id: int,
        profit: float,
        block_selection_policy: Any,
        n_blocks: int,
        budget: Budget,
    ):
        """
        A Task that requires (the same) `budget` for all blocks in `block_ids`
        """
        self.budget = budget
        super().__init__(id, profit, block_selection_policy, n_blocks)

    def set_budget_per_block(self, block_ids: Iterable[int]):
        for block_id in block_ids:
            self.budget_per_block[block_id] = self.budget
