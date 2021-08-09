from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, OrderedDict

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
DELTA = 1e-5
DPBudget = namedtuple("ConvertedDPBudget", ["epsilon", "delta", "best_alpha"])


class Budget:
    def __init__(self, orders: Dict[float, float]) -> None:
        # "Immutable" dict sorted by small alphas first
        self.__orders = {}
        for alpha in sorted(orders):
            self.__orders[alpha] = orders[alpha]

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

    def is_positive(self) -> bool:
        for epsilon in self.epsilons:
            if epsilon >= 0:
                return True
        return False

    @property
    def alphas(self) -> list:
        return list(self.__orders.keys())

    @property
    def epsilons(self) -> list:
        return list(self.__orders.values())

    def epsilon(self, alpha: float) -> float:
        return self.__orders[alpha]

    def dp_budget(self, delta: float = DELTA) -> DPBudget:
        """
        Uses a tight conversion formula to get (epsilon, delta)-DP.
        It can be slow to compute for the first time.
        """

        if hasattr(self, "dp_budget_cached"):
            return self.dp_budget_cached

        epsilon, best_alpha = get_privacy_spent(
            orders=list(self.alphas),
            rdp=list(self.epsilons),
            delta=delta,
        )
        # Cache the result
        self.dp_budget_cached = DPBudget(
            epsilon=epsilon, delta=delta, best_alpha=best_alpha
        )

        return self.dp_budget_cached

    def add_with_threshold(self, other: "Budget", threshold: "Budget"):
        """
        Increases every budget-epsilon by "amount".
        The maximum value a budget-epsilon can take is threshold-epsilon.
        """
        return Budget(
            {
                alpha: min(
                    self.epsilon(alpha) + other.epsilon(alpha), threshold.epsilon(alpha)
                )
                for alpha in self.alphas
            }
        )

    def can_allocate(self, demand_budget):
        """
        There must exist at least one order in the block's budget
        that is smaller or equal to the corresponding order of the demand budget.
        """
        diff = self - demand_budget
        max_order = max(diff.epsilons)
        if max_order >= 0:
            return True
        return False

    def __sub__(self, other):
        # TODO: Deal with range check and exceptions
        return Budget(
            {alpha: self.epsilon(alpha) - other.epsilon(alpha) for alpha in self.alphas}
        )

    def __add__(self, other):
        return Budget(
            {alpha: self.epsilon(alpha) + other.epsilon(alpha) for alpha in self.alphas}
        )

    def __truediv__(self, n: int):
        return Budget({alpha: self.epsilon(alpha) / n for alpha in self.alphas})

    def __repr__(self) -> str:
        return "Budget({})".format(self.__orders)

    def __ge__(self, other) -> bool:
        diff = self - other
        return diff.is_positive()

    def copy(self):
        return Budget(self.__orders.copy())
