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
        self.histogram_data = SparseHistogram.from_dataframe(raw_data)

    def run(query, budget, disable_dp=False):
        if isinstance(query, Tensor):
            result = self.histogram_data.run(query)
            result /= self.size
            sensitivity = 1 / self.size

        elif isinstance(query, DataFrame):
            pass

        if not disable_dp:
            noise_sample = np.random.laplace(
                scale=sensitivity / budget.epsilon
            )  # todo: this is not compatible with renyi
            result += noise_sample
        return result


# Block comprising multiple partial blocks
class HyperBlock(Block):
    def __init__(self, block_ids, blocks):
        self.id = block_ids
        self.blocks = blocks
        
        self.size = 0
        self.raw_data = []
        for block in blocks:
            self.size += block.size
            if block.raw_data:
                self.raw_data += block.raw_data
            else:
                self.raw_data += pd.read_csv(f"{block.data_path}/block_{block.id}.csv")

        # TODO: build hyperhistogram from partial histograms instead
        self.histogram_data = SparseHistogram.from_dataframe(raw_data)