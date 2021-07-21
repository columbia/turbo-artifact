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

    indices = [i for i in range(n)]

    # Tuple dict
    x = m.addVars(indices, vtype=GRB.BINARY, name="x")
    print(x)

    # Coeffs for a single alpha (demo)
    # TODO: add the OR here
    alpha = block.alphas[5]
    alpha_0 = {i: job_list[i].orders[alpha] for i in indices}

    print(f"Demands for alpha 0:{alpha_0}")
    print(f"Capacity for alpha_0: {block.orders[alpha]}")

    m.addConstr(x.prod(alpha_0) <= block.orders[alpha])

    m.setObjective(x.sum(), GRB.MAXIMIZE)
    m.optimize()

    print(x)
    print(x[0])
    print(type(x[0].x))

    return [i for i in indices if x[i].x]


def pack_one_block_matrix(job_list, block):
    """
    Returns a list of booleans corresponding to the jobs that are allocated
    """
    m = gp.Model("pack")
    n = len(job_list)
    d = len(block.alphas)

    # x[i] = 1 if job i is packed
    x = m.addMVar(shape=n, vtype=GRB.BINARY, name="x")
    max_remaining_budget = m.addVar(vtype=GRB.CONTINUOUS, name="max_remaining_budget")

    # Maximize the number of packed jobs (profit = 1)
    m.setObjective(x.sum(), GRB.MAXIMIZE)

    # Each line is one alpha, each column is one job
    demand_matrix = np.array(
        [[job.orders[alpha] for job in job_list] for alpha in block.alphas]
    )
    # Vector of consumed budgets
    consumed_budget = demand_matrix.dot(x)

    # # List of booleans indicating which alphas are still positive
    # positive_alphas = [
    #     (0 <= (block.orders[alpha] - consumed_budget[i]))
    #     for (i, alpha) in enumerate(block.alphas)
    # ]
    # print(positive_alphas)

    remaining_budget = [
        block.orders[alpha] - consumed_budget[i]
        for (i, alpha) in enumerate(block.alphas)
    ]

    # The sum of the curves should be below the capacity for one alpha
    # m.addGenConstrOr(1, positive_alphas)
    # m.addConstr(max_remaining_budget == gp.max_(remaining_budget))
    # m.addConstr(max_remaining_budget >= 0.0)

    print(remaining_budget[0])
    for i in range(d):
        m.addConstr(0.0 <= remaining_budget[i])

    m.optimize()
    print(x.X)

    return x.X


def main():
    block = Budget.from_epsilon_delta(epsilon=10, delta=0.001)

    jobs = [GaussianBudget(sigma=s) for s in np.linspace(0.1, 1, 10)]

    # random.shuffle(jobs)

    # TODO: add Laplace jobs to disturb DPF?

    # print(jobs)

    # print(block)

    print(pack_one_block(jobs, block))


if __name__ == "__main__":
    main()
