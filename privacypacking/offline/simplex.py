"""
Comparing DPF vs Packing
Offline
"""

import random

import gurobipy as gp
from gurobipy import GRB

from privacypacking.budget import ALPHAS
from privacypacking.budget.block import *
from privacypacking.budget.task import *
from privacypacking.plot import plot


def pack_blocks(task_list, blocks):
    """
    Returns a list of booleans corresponding to the tasks that are allocated
    """
    m = gp.Model("pack")
    n = len(task_list)
    d = len(ALPHAS)

    demands_upper_bound = {}
    for k, block in enumerate(blocks):
        for alpha in block.budget.alphas:
            demands_upper_bound[(k, alpha)] = sum(
                [task.budget_per_block[k].orders[alpha] for task in task_list]
            )

    # Variables
    x = m.addMVar((1, n), vtype=GRB.BINARY, name="x")
    a = m.addMVar((len(blocks), d), vtype=GRB.BINARY, name="a")
    print(x, a)

    # Constraints
    for k, _ in enumerate(blocks):
        m.addConstr(a[k].sum() >= 1)

    for k, block in enumerate(blocks):
        for i, alpha in enumerate(block.budget.alphas):
            demands = [
                task.budget_per_block[k].orders[alpha] for j, task in enumerate(task_list)
            ]
            m.addConstr(
                sum([x[0, z] * demands[z] for z in range(n)])
                - (1 - a[k, i]) * demands_upper_bound[(k, alpha)]
                <= block.budget.orders[alpha]
            )
    print(m)

    # Objective function
    m.setObjective(x.sum(), GRB.MAXIMIZE)
    m.optimize()

    return [bool((abs(x[0, i].x - 1) < 1e-4)) for i in range(n)]


def main():
    # num_blocks = 1 # single-block case
    num_blocks = 2  # multi-block case

    blocks = [create_block(i, 10, 0.001) for i in range(num_blocks)]
    tasks = ([create_gaussian_task(i, num_blocks, range(num_blocks), s) for i, s in
              enumerate(np.linspace(0.1, 1, 10))] +
             [create_gaussian_task(i, num_blocks, range(num_blocks), l) for i, l in
              enumerate(np.linspace(0.1, 10, 5))] +
             [create_subsamplegaussian_task(i, num_blocks, range(num_blocks),
                                            ds=60_000,
                                            bs=64,
                                            epochs=10,
                                            s=s) for i, s in enumerate(np.linspace(1, 10, 5))]
             )


    random.shuffle(tasks)

    allocation = pack_blocks(tasks, blocks)
    plot(tasks, blocks, allocation)

if __name__ == "__main__":
    main()
