from curses import raw
from typing import List

from privacypacking.budget import ALPHAS, Budget
from privacypacking.budget import SparseHistogram
import pandas as pd


class Block:
    def __init__(self, block_id, budget, data_path=""):
        self.id = block_id
        self.initial_budget = budget
        self.budget = budget
        self.data_path = data_path
        self.size = None
        self.raw_data = None
        self.histogram_data = None

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
        self.raw_data = pd.read_csv(f"{self.data_path}/block_{self.id}.csv")

    def load_histogram(self, attribute_domain_sizes) -> SparseHistogram:
        raw_data = self.raw_data
        if raw_data is None:
            raw_data = pd.read_csv(f"{self.data_path}/block_{self.id}.csv")

        self.bns = {}
        # raw_data.groupby(raw_data.head()).count()
        
        self.histogram_data = SparseHistogram(
            bin_indices=[(0, 0, 1), (1, 0, 5), (0, 1, 2)],
            values=[4, 1, 2],
            attribute_sizes=attribute_domain_sizes, #[2, 2, 10],
        )
        self.size = 3
