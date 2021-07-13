from typing import List

from privacypacking.budget import ALPHAS, Budget


class LaplaceBudget(Budget):
    pass


class GaussianBudget(Budget):
    def __init__(self, sigma: float, alpha_list: List[float] = ALPHAS) -> None:
        orders = {alpha: alpha / (2 * (sigma ** 2)) for alpha in alpha_list}
        super().__init__(orders)


class SubsampledGaussianBudget(Budget):
    pass
