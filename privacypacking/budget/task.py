from typing import Iterable

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
        """Tasks are assumed to be immutable: the demands won't change over time.
        The scheduler can be stateful (e.g. store whether task `id` has been scheduled).

        Args:
            id (int): unique identifier
            profit (float): how much profit/reward the task gives if it is scheduled
            budget_per_block (Dict[int,): task demand
        """
        self.id = id
        self.profit = profit
        self.block_selection_policy = block_selection_policy
        self.n_blocks = n_blocks
        # The scheduler sets the budget_per_block according to the task's block selection
        # policy and the current state of existing blocks
        # i.e. it is re-set every time the task is considered for scheduling
        # is set only once for offline setting
        # Add Other API / constraints
        # block selection policy and task_num_blocks is user-defined (API)
        self.budget_per_block = {}

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
            block_selection_policy: BlockSelectionPolicy,
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
