import itertools
import random

import numpy as np


def get_subsets_of_contiguous_blocks(x):
    subsets = []
    xlen = len(x)
    for size in range(xlen):
        for i in range(xlen - size):
            subsets.append(tuple([x[i], x[i + size]]))
    return subsets


def get_splits(x, num_cuts):
    splits = []
    xlen = len(x)
    for cuts in itertools.combinations(range(1, xlen), num_cuts):
        c = list(cuts)
        c.append(xlen)
        c.insert(0, 0)
        split = [tuple(x[c[i]: c[i + 1]]) for i in range(len(c) - 1)]
        splits.append(split)
    return splits


def normalized_absolute_mean_error(res1, res2):
    res1 = res1.to_numpy()
    res2 = res2.to_numpy()
    d = max(np.abs(res1), np.abs(res2))
    if not d:
        # print(f"Res1={res1}, Res2={res2} -     distance: 0")
        return 0
    # print(f"Res1={res1}, Res2={res2} -     distance: {np.abs(res1-res2) / d}")
    return np.abs(res1 - res2) / d


def upper_bound_normalized_absolute_mean_error(res1, res2):
    (epsilon1, result1) = res1
    (epsilon2, result2) = res2

    def calculate_noise(sensitivity, epsilon, beta):
        return (np.sqrt(2) * np.log(sensitivity/beta)) / epsilon

    def process(a):
        if a < 0:
            return 0
        return a

    f1_dp = result1.to_numpy()
    f2_dp = result2.to_numpy()

    noise1 = calculate_noise(1, epsilon1, 0.01)
    noise2 = calculate_noise(1, epsilon2, 0.01)

    f1_range = [process(f1_dp-noise1), process(f1_dp+noise1)]
    f2_range = [process(f2_dp-noise2), process(f2_dp+noise1)]

    if f1_range[1] > f2_range[0]:
        max_ = f1_range[1]
        min_ = f2_range[0]
    else:
        max_ = f2_range[1]
        min_ = f1_range[0]

    if not max_:
        print(f"Res1={res1}, Res2={res2} -     distance: 0")
        return 0

    print(f"Res1={f1_dp}, Res2={f2_dp} -     distance: {(max_ - min_) / max_}")
    return (max_ - min_) / max_