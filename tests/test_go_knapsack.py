import ctypes

import numpy as np
import numpy.ctypeslib as npct
from loguru import logger


def test_go():
    # go_knapsack = ctypes.cdll.LoadLibrary("./knapsack.so")
    go_knapsack = npct.load_library("knapsack", ".")
    solveApprox = go_knapsack.SolveApprox

    array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags="CONTIGUOUS")

    solveApprox.argtypes = [
        ctypes.c_double,
        array_1d_double,
        array_1d_double,
        ctypes.c_int,
    ]
    solveApprox.restype = ctypes.c_double

    capacity = 1
    data = np.array([0.0, 1.0, 2.0])
    profits = np.array([0.0, 1.0, 2.0])

    c = solveApprox(capacity, data, profits, len(data))
    print(c)


if __name__ == "__main__":
    test_go()
