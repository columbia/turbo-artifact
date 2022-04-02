from collections import defaultdict

import numpy as np
import pandas as pd
import scipy

from privacypacking.budget import Budget
from privacypacking.budget.curves import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
    SubsampledLaplaceCurve,
)


def build_zoo() -> list:
    curve_zoo = []
    task_names = []

    for sigma in np.geomspace(0.01, 10, 100):
        # for sigma in np.linspace(0.01, 100, 100):
        curve_zoo.append(GaussianCurve(sigma=sigma))
        task_names.append(f"gaussian-{sigma:.4f}")
    for sigma in np.geomspace(0.01, 10, 100):
        curve_zoo.append(LaplaceCurve(laplace_noise=sigma))
        task_names.append(f"laplace-{sigma:.4f}")

    for sigma in np.geomspace(0.01, 10, 5):
        # for sigma in np.linspace(0.01, 100, 100):

        for sampling in np.geomspace(1e-5, 0.5, 5):
            for steps in [1] + [200 * k for k in range(1, 5)]:
                curve_zoo.append(
                    SubsampledGaussianCurve(
                        sigma=sigma, sampling_probability=sampling, steps=steps
                    )
                )
                task_names.append(
                    f"subsampledgaussian-{sigma:.4f}_{sampling:.6f}_{steps}"
                )

                curve_zoo.append(
                    SubsampledLaplaceCurve(
                        noise_multiplier=sigma,
                        sampling_probability=sampling,
                        steps=steps,
                    )
                )
                task_names.append(
                    f"subsampledlaplace-{sigma:.4f}_{sampling:.6f}_{steps}"
                )

    return list(zip(task_names, curve_zoo))


def zoo_df(zoo: list, clipped=True, epsilon=10, delta=1e-8) -> pd.DataFrame:
    block = Budget.from_epsilon_delta(epsilon=10, delta=1e-8)

    dict_list = defaultdict(list)
    for index, name_and_curve in enumerate(zoo):
        name, curve = name_and_curve
        for alpha, epsilon in zip(curve.alphas, curve.epsilons):
            if block.epsilon(alpha) > 0:
                dict_list["alphas"].append(alpha)
                dict_list["rdp_epsilons"].append(epsilon)
                if clipped:
                    dict_list["normalized_epsilons"].append(
                        min(epsilon / block.epsilon(alpha), 1)
                    )
                else:
                    dict_list["normalized_epsilons"].append(
                        epsilon / block.epsilon(alpha)
                    )
                dict_list["task_id"].append(index)
                dict_list["task_name"].append(name)
    df = pd.DataFrame(dict_list)

    tasks = pd.DataFrame(
        df.groupby("task_id")["normalized_epsilons"].agg(min)
    ).reset_index()
    tasks = tasks.rename(columns={"normalized_epsilons": "epsilon_min"})
    tasks["epsilon_max"] = df.groupby("task_id")["normalized_epsilons"].agg(max)
    tasks["epsilon_range"] = tasks["epsilon_max"] - tasks["epsilon_min"]
    tasks = tasks.query("epsilon_min < 1 and epsilon_min > 1e-3")

    df = df.merge(tasks, on="task_id")

    return df, tasks


def geometric_frequencies(tasks_df: pd.DataFrame, n_bins=20, p=0.5) -> pd.DataFrame:
    def map_range_to_bin(r):
        return int(r * n_bins)

    df = tasks_df.copy()
    df["bin_id"] = df["epsilon_range"].apply(map_range_to_bin)

    count_by_bin = list(df.groupby("bin_id").count().epsilon_range)

    def map_bin_to_freq(k):
        # Geometric distribution to choose the bin, then uniformly at random inside each bin
        return (1 - p) ** (k - 1) * p / count_by_bin[k]

    df["frequency"] = df["bin_id"].apply(map_bin_to_freq)
    # We normalize (we chopped off the last bins, + some error is possible)
    df["frequency"] = df["frequency"] / df["frequency"].sum()

    return df


def gaussian_block_distribution(mu, sigma, max_blocks):

    if sigma == 0:
        return f"{mu}:1"

    f = []
    for k in range(1, max_blocks + 1):
        f.append(
            scipy.stats.norm.pdf(k, mu, sigma)
            # scipy.special.binom(k + r - 1, k) * (1-p)**r * p**k
            # k **(alpha - 1) * np.exp(-beta * k) * beta ** alpha / scipy.special.gamma(alpha)
        )
    f = np.array(f)
    f = f / sum(f)

    name_and_freq = []
    for k, freq in enumerate(f):
        name_and_freq.append(f"{k+1}:{float(freq)}")

    return ",".join(name_and_freq)
