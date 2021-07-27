from collections import namedtuple
from typing import Dict, List, NamedTuple

import numpy as np
from opacus.privacy_analysis import get_privacy_spent

# TODO: other range of default alphas?
ALPHAS = [
    1.5,
    1.75,
    2,
    2.5,
    3,
    4,
    5,
    6,
    8,
    16,
    32,
    64,
]  # , 1e6] omitting last alpha for better visualization

# Default value for MNIST-like delta
DELTA = 1e-6
DPBudget = namedtuple("ConvertedDPBudget", ["epsilon", "delta", "best_alpha"])


# TODO: make it immutable to remove ambiguity?
# And blocks can have a mutable budget field.


class Budget:
    def __init__(self, orders: Dict[float, float]) -> None:
        # TODO: float or other type? Floating point arith
        # TODO: sorted dict? And keep it sorted.
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

    @property
    def alphas(self) -> list:
        return list(self.orders.keys())

    def dp_budget(self, delta: float = DELTA) -> DPBudget:
        """
        Uses a tight conversion formula to get (epsilon, delta)-DP.
        It can be slow to compute for the first time.
        """

        epsilon, best_alpha = get_privacy_spent(
            orders=list(self.orders.keys()),
            rdp=list(self.orders.values()),
            delta=delta,
        )
        # TODO: cache this? If orders is immutable.

        return DPBudget(epsilon=epsilon, delta=delta, best_alpha=best_alpha)
