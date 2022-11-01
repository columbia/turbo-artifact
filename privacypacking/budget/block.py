from typing import List, Dict
from privacypacking.budget import ALPHAS, Budget, SparseHistogram
from torch import Tensor
import pandas as pd
import numpy as np


class Block:
    def __init__(self, block_id, budget):
        self.id = block_id
        self.initial_budget = budget
        self.budget = budget
        self.size = None
        self.raw_data = None
        self.histogram_data = None
        self.domain_size= None
        self.data_path = None

    @classmethod
    def from_epsilon_delta(
        cls,
        block_id: int,
        epsilon: float,
        delta: float,
        alpha_list: List[float] = ALPHAS,
    ) -> "Block":
        return cls(
            block_id,
            Budget.from_epsilon_delta(
                epsilon=epsilon, delta=delta, alpha_list=alpha_list
            ),
        )

    def dump(self):
        return {
            "id": self.id,
            "initial_budget": self.initial_budget.dump(),
            "budget": self.budget.dump(),
            "data_path": self.data_path,
        }

    def __len__(self) -> int:
        return self.size

    @property
    def remaining_budget(self) -> Budget:
        return self.budget

    @property
    def allocated_budget(self) -> Budget:
        return self.initial_budget - self.budget

    @property
    def is_exhausted(self) -> bool:
        return self.remaining_budget.is_exhausted()

    # For compatibility with online schedulers
    @property
    def available_unlocked_budget(self) -> Budget:
        return self.budget

    def load_raw_data(self,):
        self.raw_data = pd.read_csv(self.data_path)

    def load_histogram(self, attribute_domain_sizes) -> SparseHistogram:
        raw_data = self.raw_data
        if raw_data is None:
            raw_data = pd.read_csv(self.data_path)
        self.histogram_data = SparseHistogram.from_dataframe(
            raw_data, attribute_domain_sizes
        )

    def run(self, query):
        if isinstance(query, Tensor):
            result = self.histogram_data.run(query)
            result /= self.size
        elif isinstance(query, pd.DataFrame):
            pass
        return result

    def run_dp(self, query, budget):
        result = self.run(query)
        sensitivity = 1 / self.size
        noise_sample = np.random.laplace(
            scale=sensitivity / budget.epsilon
        )  # TODO: this is not compatible with renyi
        result += noise_sample
        return result


# Wraps one or more blocks
class HyperBlock:
    def __init__(self, blocks: Dict):
        self.blocks = blocks
        block_ids = list(blocks.keys())
        self.id = (block_ids[0], block_ids[-1])
        self.size = 0

        for block in self.blocks.values():
            self.size += len(block)
        
        self.domain_size = 0
        if block_ids:
            self.domain_size = self.blocks[block_ids[0]].domain_size

    def run(self, query):
        if isinstance(query, Tensor):
            # Weighted average of dot products
            result = 0
            for block in self.blocks.values():
                result += len(block) * block.run(query)
            result /= self.size
        elif isinstance(query, pd.DataFrame):
            pass
        return result

    def run_dp(self, query, budget):
        result = self.run(query)
        sensitivity = 1 / self.size
        noise_sample = np.random.laplace(
            scale=sensitivity / budget.epsilon
        )  # TODO: this is not compatible with renyi
        result += noise_sample
        return result

    def can_run(self, demand) -> bool:
        """
        A task can run only if we can allocate the demand budget
        for all the blocks requested
        """
        for block_id, demand_budget in demand.items():
            if block_id not in self.blocks:
                return False
            block = self.blocks[block_id]

            if not block.budget.can_allocate(demand_budget):
                return False
        return True
