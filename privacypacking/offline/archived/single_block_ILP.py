"""
Comparing DPF vs Packing
Offline, single block setting
"""

import random

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from privacypacking.budget import Budget
from privacypacking.budget.block import create_blocks

from privacypacking.budget.curves import (
    GaussianBudget,
    LaplaceBudget,
    SubsampledGaussianBudget,
)
from privacypacking.plot import singleplot


def check_same_support():
    pass


def pack_one_block(job_list, block):
    """
    Returns a list of booleans corresponding to the jobs that are allocated
    """
    m = gp.Model("pack")
    n = len(job_list)
    d = len(block.alphas)

    demands_upper_bound = {}
    for alpha in block.alphas:
        demands_upper_bound[alpha] = sum([job.orders[alpha] for job in job_list])

    # Variables
    x = m.addVars(n, vtype=GRB.BINARY, name="x")
    a = m.addVars(d, vtype=GRB.BINARY, name="a")
    print(x, a)

    # Constraints
    m.addConstr(a.sum() >= 1)
    for i, alpha in enumerate(block.alphas):
        demands = {j: job.orders[alpha] for j, job in enumerate(job_list)}
        m.addConstr(
            x.prod(demands) - (1 - a[i]) * demands_upper_bound[alpha]
            <= block.orders[alpha]
        )
        print(m)

    # Objective function
    m.setObjective(x.sum(), GRB.MAXIMIZE)
    m.optimize()

    return [(abs(x[i].x - 1) < 1e-4) for i in range(n)]


def main():
    block = Budget.from_epsilon_delta(epsilon=10, delta=0.001)

    jobs = (
        [GaussianBudget(sigma=s) for s in np.linspace(1, 10, 5)]
        + [LaplaceBudget(laplace_noise=l) for l in np.linspace(0.1, 10, 5)]
        + [
            SubsampledGaussianBudget.from_training_parameters(
                dataset_size=60_000,
                batch_size=64,
                epochs=10,
                sigma=s,
            )
            for s in np.linspace(1, 10, 5)
        ]
    )

    random.shuffle(jobs)

    allocation = pack_one_block(jobs, block)
    singleplot(jobs, block, allocation)


if __name__ == "__main__":
    main()
