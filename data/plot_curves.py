from privacypacking.utils import (
    compute_laplace_task_demands,
    compute_gaussian_task_demands_from_sigma,
    compute_subsampled_gaussian_task_demands,
)
import pandas as pd

alphas = [
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
]

df = pd.DataFrame()

sigma_range = [0.1, 0.2, 0.3, 1, 2, 3]
for i, sigma in enumerate(sigma_range):
    epsilons = compute_laplace_task_demands.compute_laplace_demands(sigma)
    df["curve"].append("gaussian")
    df["sigma"].append("sigma")
    df["alpha"].append(alphas)
    df["epsilons"].append(epsilons)

sigma_range = [0.1, 0.2, 0.3, 1, 2, 3]
for i, sigma in enumerate(sigma_range):
    epsilons = compute_gaussian_task_demands_from_sigma.compute_gaussian_demands(sigma)
    df["curve"].append("laplace")
    df["sigma"].append("sigma")
    df["alpha"].append(alphas)
    df["epsilons"].append(epsilons)

sigma_range = [0.1, 0.2, 0.3, 1, 2, 3]
for i, sigma in enumerate(sigma_range):
    epsilons = compute_subsampled_gaussian_task_demands.compute_subsampled_gaussian_task_demands(
        sigma
    )
    df["curve"].append("subsampled")
    df["sigma"].append("sigma")
    df["alpha"].append(alphas)
    df["epsilons"].append(epsilons)
