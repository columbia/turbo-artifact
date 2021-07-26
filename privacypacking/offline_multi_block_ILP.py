"""
Comparing DPF vs Packing
Offline, multi block setting
"""

import random

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from privacypacking.budget import Budget, ALPHAS
from privacypacking.curves import MultiblockGaussianBudget
from privacypacking.plot import save_fig,  multiplot


def pack_many_blocks(job_list, blocks):

    """
    Returns a list of booleans corresponding to the jobs that are allocated
    """
    m = gp.Model("pack")
    n = len(job_list)
    d = len(ALPHAS)

    demands_upper_bound = {}
    for k, block in enumerate(blocks):
        for alpha in block.alphas:
            demands_upper_bound[(k,alpha)] = sum([job.block_budgets[k].orders[alpha] for job in job_list])

    # Variables
    x = m.addMVar((1,n), vtype=GRB.BINARY, name="x")
    a = m.addMVar((len(blocks),d), vtype=GRB.BINARY, name="a")
    print(x, a)

    # Constraints
    for k, _ in enumerate(blocks):
        print(k)
        m.addConstr(a[k].sum() >= 1)

    for k, block in enumerate(blocks):
        for i, alpha in enumerate(block.alphas):
            print(demands_upper_bound[(k,alpha)])
            print((1-a[k][i])*demands_upper_bound[(k,alpha)])
            demands = [job.block_budgets[k].orders[alpha] for j, job in enumerate(job_list)]
            m.addConstr(sum([x[0,z]*demands[z] for z in range(n)]) - (1-a[k,i])*demands_upper_bound[(k,alpha)] <= block.orders[alpha])
    print(m)

    # Objective function
    m.setObjective(x.sum(), GRB.MAXIMIZE)
    m.optimize()

    return [bool((abs(x[0,i].x - 1) < 1e-4)) for i in range(n)]


def main():
    num_blocks = 2
    blocks = []
    for _ in range(num_blocks):
        blocks += [Budget.from_epsilon_delta(epsilon=10, delta=0.001)]

    jobs = [MultiblockGaussianBudget(num_blocks, sigma=s) for s in np.linspace(0.1, 1, 10)]

    random.shuffle(jobs)

    allocation = pack_many_blocks(jobs, blocks)
    print(allocation)
    multiplot(jobs, blocks, allocation)


if __name__ == "__main__":
    main()
