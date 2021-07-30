"""
Comparing DPF vs Packing
Offline, single block setting
"""

import random

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from privacypacking.budget import Budget
from privacypacking.budget.curves import GaussianBudget
from privacypacking.plot import singleplot


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


def main():
    block = Budget.from_epsilon_delta(epsilon=10, delta=0.001)

    jobs = [GaussianBudget(sigma=s) for s in np.linspace(0.1, 1, 10)]

    random.shuffle(jobs)

    # TODO: add Laplace jobs to disturb DPF?

    allocation = pack_one_block(jobs, block)

    singleplot(jobs, block, allocation)


if __name__ == "__main__":
    main()
