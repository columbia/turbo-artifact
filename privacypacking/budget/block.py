from typing import List

from privacypacking.budget import Budget


class Block:
    def __init__(self, id, budget):
        self.id = id
        self.initial_budget = budget
        self.budget = budget
        # add other properties here

    @classmethod
    def from_epsilon_delta(
        cls, block_id: int, epsilon: float, delta: float, alpha_list: List[float]
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

    @property
    def remaining_budget(self) -> Budget:
        return self.budget

    @property
    def allocated_budget(self) -> Budget:
        return self.initial_budget - self.budget

    # For compatibility with online schedulers
    @property
    def available_unlocked_budget(self) -> Budget:
        return self.budget
