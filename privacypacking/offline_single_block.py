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


def toy():
    """Dealing with 'exists' or max >=0 with Gurobi"""
    m = gp.Model("pack")

    a1 = 1.001
    a2 = 1.5

    b1 = 0.5
    b2 = 1.1

    c1 = 1.0
    c2 = 1.0

    xa = m.addVar(vtype=GRB.BINARY, name="xa")
    xb = m.addVar(vtype=GRB.BINARY, name="xb")

    # m.addConstr(1 == gp.or_(a1 * xa + b1 * xb <= c1, a2 * xa + b2 * xb <= c2))
    # gurobipy.GurobiError: General expressions can only be the right hand side part of an assignment to a single variable

    # m.addConstr(0.0 <= gp.max_([c1 - a1 * xa + b1 * xb, c2 - a2 * xa + b2 * xb]))
    # TypeError: '<=' not supported between instances of 'float' and 'GenExpr'

    # z = m.addVar(vtype=GRB.CONTINUOUS, name="z")
    # m.addConstr(z == gp.max_([c1 - (a1 * xa + b1 * xb), c2 - (a2 * xa + b2 * xb)]))
    # m.addConstr(z >= 0.0)
    # gurobipy.GurobiError: Invalid data in vars array

    z = m.addVar(vtype=GRB.CONTINUOUS, name="z")
    s1 = m.addVar(lb=-float("inf"), ub=float("inf"), vtype=GRB.CONTINUOUS, name="s1")
    s2 = m.addVar(lb=-float("inf"), ub=float("inf"), vtype=GRB.CONTINUOUS, name="s2")

    m.addConstr(s1 == c1 - ((a1 * xa) + (b1 * xb)))
    m.addConstr(s2 == c2 - ((a2 * xa) + (b2 * xb)))

    m.addConstr(z == gp.max_([s1, s2]))
    m.addConstr(z >= 0.0)

    m.setObjective(xa + xb, GRB.MAXIMIZE)
    print(m)
    m.optimize()

    print(s1.x, s2.x)

    return xa.x, xb.x


def pack_one_block(job_list, block):

    """
    Returns a list of booleans corresponding to the jobs that are allocated
    """
    m = gp.Model("pack")
    n = len(job_list)
    d = len(block.alphas)

    x = m.addVars(n, vtype=GRB.BINARY, name="x")
    s = m.addVars(d, lb=-float("inf"), ub=float("inf"), vtype=GRB.CONTINUOUS, name="s")
    z = m.addVar(vtype=GRB.CONTINUOUS, name="z")
    print(x, s, z)

    for i, alpha in enumerate(block.alphas):
        demands = {j: job.orders[alpha] for j, job in enumerate(job_list)}
        m.addConstr(s[i] == block.orders[alpha] - x.prod(demands))
        print(m)

    m.addConstr(z == gp.max_(s))
    m.addConstr(z >= 0.0)

    m.setObjective(x.sum(), GRB.MAXIMIZE)
    m.optimize()

    return [(abs(x[i].x - 1) < 1e-4) for i in range(n)]


def pack_one_block_one_alpha(job_list, block):
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

    random.shuffle(jobs)

    # TODO: add Laplace jobs to disturb DPF?

    # print(jobs)

    # print(block)

    allocation = pack_one_block(jobs, block)

    stack_jobs_under_block_curve(jobs, block, allocation)
    # print(toy())


if __name__ == "__main__":
    main()
