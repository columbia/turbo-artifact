from typing import Any, Iterable

import numpy as np
from scipy.sparse import bsr_matrix, dok_matrix
from scipy.sparse.construct import vstack

from privacypacking.budget.block_selection import BlockSelectionPolicy
from privacypacking.budget.budget import ALPHAS, Budget
from privacypacking.budget.curves import ZeroCurve


class Task:
    def __init__(
        self,
        id: int,
        query_id: int,
        query_type: str,
        profit: float,
        block_selection_policy: BlockSelectionPolicy,
        n_blocks: int,
        name: str = None,
    ):
        # User request
        self.id = id
        self.query_id = query_id
        self.query_type = query_type
        self.profit = profit
        self.block_selection_policy = block_selection_policy
        self.n_blocks = n_blocks
        self.name = name
        # Scheduler dynamically updates the variables below
        self.budget_per_block = {}
        self.initial_budget_per_block = {}
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

    def get_substitute_demand(self, substitute):
        pass

    def dump(self):
        return {
            "id": self.id,
            "query_id": self.query_id,
            "profit": self.profit,
            "start_time": None,
            "allocation_time": None,
            "budget_per_block": {
                block_id: budget.dump()
                for block_id, budget in self.budget_per_block.items()
            },
        }

    def build_demand_matrix(self, alphas=ALPHAS, max_block_id=None):
        # Prepare a sparse matrix of the demand
        max_block_id = max_block_id or max(self.budget_per_block.keys())
        n_alphas = len(alphas)

        # NOTE: Using a dumb matrix is faster, go back to sparse if we have RAM issues.
        self.demand_matrix = np.zeros((max_block_id + 1, n_alphas))
        for block_id, budget in self.budget_per_block.items():
            for i, alpha in enumerate(alphas):
                self.demand_matrix[block_id, i] = budget.epsilon(alpha)

        #
        # temp_matrix = dok_matrix((max_block_id + 1, n_alphas))
        # for block_id, budget in self.budget_per_block.items():
        #     for i, alpha in enumerate(alphas):
        #         temp_matrix[block_id, i] = budget.epsilon(alpha)

        # Block compressed matrix, since we have chunks of non-zero values
        # self.demand_matrix = bsr_matrix(temp_matrix)
        # self.demand_matrix = temp_matrix.toarray()

    # def pad_demand_matrix(self, n_blocks, alphas=ALPHAS):
    #     if not hasattr(self, "demand_matrix"):
    #         self.build_demand_matrix(alphas)
    #     n_new_rows = n_blocks - self.demand_matrix.shape[0]
    #     n_columns = self.demand_matrix.shape[1]

    #     self.demand_matrix = vstack(
    #         [self.demand_matrix, bsr_matrix((n_new_rows, n_columns))]
    #     )


class UniformTask(Task):
    def __init__(
        self,
        id: int,
        query_id: int,
        query_type: str,
        profit: float,
        block_selection_policy: Any,
        n_blocks: int,
        budget: Budget,
        name: str = None,
    ):
        """
        A Task that requires (the same) `budget` for all blocks in `block_ids`
        """
        self.budget = budget
        super().__init__(id, query_id, query_type, profit, block_selection_policy, n_blocks, name=name)

    def set_budget_per_block(self, block_ids: Iterable[int]):
        for block_id in block_ids:
            self.budget_per_block[block_id] = self.budget
            self.initial_budget_per_block[block_id] = self.budget