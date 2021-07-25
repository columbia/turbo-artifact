"""
Comparing DPF vs Packing
Offline, single block setting
"""

import random

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from privacypacking.budget import Budget
from privacypacking.curves import GaussianBudget
from privacypacking.plot import save_fig, stack_jobs_under_block_curve

# TODO: reduce to same support/utils/throw appropriate error


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
        m.addConstr(x.prod(demands) - (1-a[i])*demands_upper_bound[alpha] <= block.orders[alpha])
        print(m)

    # Objective function
    m.setObjective(x.sum(), GRB.MAXIMIZE)
    m.optimize()

    return [(abs(x[i].x - 1) < 1e-4) for i in range(n)]


def main():
    block = Budget.from_epsilon_delta(epsilon=10, delta=0.001)

    jobs = [GaussianBudget(sigma=s) for s in np.linspace(0.1, 1, 10)]

    random.shuffle(jobs)

    # TODO: add Laplace jobs to disturb DPF?

    allocation = pack_one_block(jobs, block)
    # print(allocation)
    stack_jobs_under_block_curve(jobs, block, allocation)


if __name__ == "__main__":
    main()
