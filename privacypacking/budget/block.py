from privacypacking.budget import Budget


class Block:
    def __init__(self, id, budget):
        self.id = id
        self.initial_budget = budget
        self.budget = budget
        # add other properties here

    @classmethod
    def from_epsilon_delta(cls, block_id: int, epsilon: float, delta: float) -> "Block":
        return cls(block_id, Budget.from_epsilon_delta(epsilon=epsilon, delta=delta))

    def dump(self):
        return {
            "id": self.id,
            "initial_budget": self.initial_budget.dump(),
            "budget": self.budget.dump(),
        }
