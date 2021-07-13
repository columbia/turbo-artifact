from typing import Dict, List

import numpy as np

# TODO: other range of default alphas?
ALPHAS = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 1e6]


class Budget:
    def __init__(self, orders: Dict[float, float]) -> None:
        # TODO: float or other type? Floating point arith
        self.orders = orders

    @classmethod
    def from_epsilon_list(
        cls, epsilon_list: List[float], alpha_list: List[float] = ALPHAS
    ) -> "Budget":

        if len(alpha_list) != len(epsilon_list):
            raise ValueError("epsilon_list and alpha_list should have the same length")

        orders = {alpha: epsilon for alpha, epsilon in zip(alpha_list, epsilon_list)}

        return cls(orders)

    @classmethod
    def from_epsilon_delta(
        cls, epsilon: float, delta: float, alpha_list: List[float] = ALPHAS
    ) -> "Budget":
        orders = {}
        for alpha in alpha_list:
            orders[alpha] = max(epsilon + np.log(delta) / (alpha - 1), 0)
        return cls(orders)

    def __sub__(self, other):
        # TODO: Deal with range check and exceptions
        return Budget(
            {alpha: self.orders[alpha] - other.orders[alpha] for alpha in self.orders}
        )

    def __add__(self, other):
        return Budget(
            {alpha: self.orders[alpha] + other.orders[alpha] for alpha in self.orders}
        )

    def __repr__(self) -> str:
        return "Budget({})".format(self.orders)

    # TODO: better semantics, other comparison utilities? Overload __gt__ & cie?
    def is_positive(self) -> bool:
        return any(self.orders.values())

    # TODO: plotting utilities
