import itertools
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
        split = [tuple(x[c[i] : c[i + 1]]) for i in range(len(c) - 1)]
        splits.append(split)
    return splits


def absolute_mean_error(res1, res2):
    res1 = res1.to_numpy()
    res2 = res2.to_numpy()
    d = max(np.abs(res1), np.abs(res2))
    if not d:
        # print(f"Res1={res1}, Res2={res2} -     distance: 0")
        return 0
    # print(f"Res1={res1}, Res2={res2} -     distance: {np.abs(res1-res2) / d}")
    return np.abs(res1 - res2) / d


def have_enough_budget(blocks, demand, sys_blocks):
    for b in range(blocks[0], blocks[1] + 1):
        if not (sys_blocks[b].remaining_budget - demand).is_positive():
            return False
    return True
