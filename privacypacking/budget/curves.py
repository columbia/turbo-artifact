from typing import List

import numpy as np

# from opacus.accountants.analysis.rdp import compute_rdp
from scipy.interpolate import interp1d, splev, splrep

# from autodp.mechanism_zoo import LaplaceMechanism
# from autodp.transformer_zoo import AmplificationBySampling
from privacypacking.budget import ALPHAS, Budget


class ZeroCurve(Budget):
    def __init__(self, alpha_list: List[float] = ALPHAS) -> None:
        orders = {alpha: 0 for alpha in alpha_list}
        super().__init__(orders)


class SyntheticPolynomialCurve(Budget):
    def __init__(
        self,
        best_alpha,
        epsilon_min,
        epsilon_left,
        epsilon_right,
        alpha_list: List[float] = ALPHAS,
        block_epsilon=10,
        block_delta=1e-8,
    ) -> None:
        def lagrange_3(x):
            x_0 = alpha_list[0]
            x_2 = alpha_list[-1]
            x_1 = best_alpha
            return (
                epsilon_left * (x - x_1) * (x - x_2) / (x_0 - x_1) * (x_0 - x_2)
                + epsilon_min * (x - x_0) * (x - x_2) / (x_1 - x_0) * (x_1 - x_2)
                + epsilon_right * (x - x_0) * (x - x_1) / (x_2 - x_0) * (x_2 - x_1)
            )

        # if best_alpha not in [epsilon_left, epsilon_right]:
        #     orders = {alpha: lagrange_3(alpha) for alpha in alpha_list}

        block = Budget.from_epsilon_delta(epsilon=block_epsilon, delta=block_delta)

        non_zero_alphas = [alpha for alpha in block.alphas if block.epsilon(alpha) > 0]
        zero_alphas = [alpha for alpha in block.alphas if block.epsilon(alpha) == 0]

        # x = [non_zero_alphas[0], best_alpha, non_zero_alphas[-2], non_zero_alphas[-1]]
        # y = [
        #     epsilon_left,
        #     epsilon_min,
        #     (epsilon_min + epsilon_right) / 2,
        #     epsilon_right,
        # ]

        # print(x, y)
        # spl = splrep(x, y, k=3)

        # rdp_epsilons = splev(non_zero_alphas, spl)

        # orders = {
        #     alpha: epsilon for alpha, epsilon in zip(non_zero_alphas, rdp_epsilons)
        # }
        x = [non_zero_alphas[0], best_alpha, non_zero_alphas[-1]]
        y = [
            epsilon_left,
            epsilon_min,
            epsilon_right,
        ]
        f = interp1d(x=x, y=y, kind="slinear")
        orders = {alpha: f(alpha) * block.epsilon(alpha) for alpha in non_zero_alphas}
        for alpha in zero_alphas:
            orders[alpha] = 1
        super().__init__(orders)


class LaplaceCurve(Budget):
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
            with np.errstate(over="raise", under="raise"):
                try:
                    ε = (1 / (α - 1)) * np.log(
                        (α / (2 * α - 1)) * np.exp((α - 1) / λ)
                        + ((α - 1) / (2 * α - 1)) * np.exp(-α / λ)
                    )
                except FloatingPointError:
                    # It means that alpha/lambda is too large (under or overflow)
                    # We just drop the negative exponential (≃0) and simplify the log
                    ε = (1 / (α - 1)) * (np.log(α / (2 * α - 1)) + (α - 1) / λ)

                orders[α] = float(ε)
        super().__init__(orders)


class GaussianCurve(Budget):
    def __init__(self, sigma: float, alpha_list: List[float] = ALPHAS) -> None:
        orders = {alpha: alpha / (2 * (sigma ** 2)) for alpha in alpha_list}
        super().__init__(orders)


# class SubsampledGaussianCurve(Budget):
#     def __init__(
#         self,
#         sampling_probability: float,
#         sigma: float,
#         steps: float,
#         alpha_list: List[float] = ALPHAS,
#     ) -> None:
#         rdp = compute_rdp(
#             q=sampling_probability,
#             noise_multiplier=sigma,
#             steps=steps,
#             orders=alpha_list,
#         )

#         orders = {alpha: epsilon for (alpha, epsilon) in zip(alpha_list, rdp)}
#         super().__init__(orders)

#     @classmethod
#     def from_training_parameters(
#         cls,
#         dataset_size: int,
#         batch_size: int,
#         epochs: int,
#         sigma: float,
#         alpha_list: List[float] = ALPHAS,
#     ) -> "SubsampledGaussianCurve":
#         """Helper function to build the SGM curve with more intuitive parameters."""

#         sampling_probability = batch_size / dataset_size
#         steps = (dataset_size * epochs) // batch_size
#         return cls(sampling_probability, sigma, steps, alpha_list)


# class SubsampledLaplaceCurve(Budget):
#     def __init__(
#         self,
#         sampling_probability: float,
#         noise_multiplier: float,
#         steps: int,
#         alpha_list: List[float] = ALPHAS,
#     ) -> None:
#
#         curve = AmplificationBySampling(PoissonSampling=True)(
#             LaplaceMechanism(b=noise_multiplier), sampling_probability
#         )
#
#         orders = {alpha: curve.get_RDP(alpha) * steps for alpha in alpha_list}
#         super().__init__(orders)
