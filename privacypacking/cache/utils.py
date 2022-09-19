import itertools
import random
import numpy as np


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