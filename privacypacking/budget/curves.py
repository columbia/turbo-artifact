from typing import List

import numpy as np
from opacus.privacy_analysis import compute_rdp

from privacypacking.budget import ALPHAS, Budget


class LaplaceBudget(Budget):
    """
    RDP curve for a Laplace mechanism with sensitivity 1.
    """

    def __init__(self, laplace_noise: float, alpha_list: List[float] = ALPHAS) -> None:
        """Computes the Laplace RDP curve.
            See Table II of the RDP paper (https://arxiv.org/pdf/1702.07476.pdf)

        Args:
            laplace_noise (float): lambda
            alpha_list (List[float], optional): RDP orders. Defaults to ALPHAS.
        """
        orders = {}
        λ = laplace_noise
        for α in alpha_list:
            ε = (1 / (α - 1)) * np.log(
                (α / (2 * α - 1)) * np.exp((α - 1) / λ)
                + ((α - 1) / (2 * α - 1)) * np.exp(-α / λ)
            )
            orders[α] = float(ε)
        super().__init__(orders)


class GaussianBudget(Budget):
    def __init__(self, sigma: float, alpha_list: List[float] = ALPHAS) -> None:
        orders = {alpha: alpha / (2 * (sigma ** 2)) for alpha in alpha_list}
        super().__init__(orders)


# TODO: let's add another level of abstraction for blocks and streams of blocks, with
# budget as a black box, so we don't have to write the multiblock version for each curve
class MultiblockGaussianBudget:
    def __init__(
        self, num_blocks, sigma: float, alpha_list: List[float] = ALPHAS
    ) -> None:

        self.block_budgets = []
        for _ in range(num_blocks):
            orders = {alpha: alpha / (2 * (sigma ** 2)) for alpha in alpha_list}
            self.block_budgets += [Budget(orders)]
            # super().__init__(orders)


class SubsampledGaussianBudget(Budget):
    def __init__(
        self,
        sampling_probability: float,
        sigma: float,
        steps: float,
        alpha_list: List[float] = ALPHAS,
    ) -> None:

        rdp = compute_rdp(
            q=sampling_probability,
            noise_multiplier=sigma,
            steps=steps,
            orders=alpha_list,
        )

        orders = {alpha: epsilon for (alpha, epsilon) in zip(alpha_list, rdp)}
        super().__init__(orders)

    @classmethod
    def from_training_parameters(
        cls,
        dataset_size: int,
        batch_size: int,
        epochs: int,
        sigma: float,
        alpha_list: List[float] = ALPHAS,
    ) -> "SubsampledGaussianBudget":
        """Helper function to build the SGM curve with more intuitive parameters."""

        sampling_probability = batch_size / dataset_size
        steps = (dataset_size * epochs) // batch_size
        return cls(sampling_probability, sigma, steps, alpha_list)
