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
    (epsilon1, result1) = res1
    (epsilon2, result2) = res2

    result1 = result1.to_numpy()
    result2 = result2.to_numpy()

    d = max(np.abs(result1), np.abs(result2))
    if not d:
        # print(f"Res1={res1}, Res2={res2} -     distance: 0")
        return 0
    # print(f"Res1={res1}, Res2={res2} -     distance: {np.abs(res1-res2) / d}")
    return np.abs(result1 - result2) / d


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
    print(f"Res 1 {f1_dp}, Res 2 {f2_dp}")
    # s1 = np.random.laplace(f1_dp, 1/epsilon1)
    # s2 = np.random.laplace(f2_dp, 1/epsilon2)
    # print(f"DP Res 1 {s1}, DP Res 2 {s2}")
    noise1 = calculate_noise(1, epsilon1, 0.01)
    noise2 = calculate_noise(1, epsilon2, 0.01)
    print(f"Noise 1 {noise1}, Noise 2 {noise2}")
    f1_range = [process(f1_dp-noise1), process(f1_dp+noise1)]
    f2_range = [process(f2_dp-noise2), process(f2_dp+noise1)]
    print(f"Range 1 {f1_range}, Range 2 {f2_range}")
    # print(f"DP Range 1 {[process(s1-noise1), process(s1+noise1)]}, DP Range 2 { [process(s2-noise2), process(s2+noise1)]}")

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
