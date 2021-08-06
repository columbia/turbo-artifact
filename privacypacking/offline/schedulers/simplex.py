import random

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from privacypacking.budget import ALPHAS, Block
from privacypacking.budget.task import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
    UniformTask,
)
from privacypacking.offline.schedulers.scheduler import Scheduler


class Simplex(Scheduler):
    def __init__(self, tasks, blocks, config=None):
        super().__init__(tasks, blocks, config)

    def schedule(self):

        """
        Returns a list of booleans corresponding to the tasks that are allocated
        """
        m = gp.Model("pack")
        n = len(self.tasks)
        d = len(ALPHAS)

        demands_upper_bound = {}
        for k, block in self.blocks.items():
            for alpha in block.budget.alphas:
                demands_upper_bound[(k, alpha)] = 0
                for task in self.tasks:
                    if k in task.budget_per_block:
                        demands_upper_bound[(k, alpha)] += task.budget_per_block[
                            k
                        ].epsilon(alpha)

        # Variables
        x = m.addMVar((1, n), vtype=GRB.BINARY, name="x")
        a = m.addMVar((len(self.blocks), d), vtype=GRB.BINARY, name="a")
        print(x, a)

        # Constraints
        for k, _ in enumerate(self.blocks):
            m.addConstr(a[k].sum() >= 1)

        for k, block in self.blocks.items():
            for i, alpha in enumerate(block.budget.alphas):
                demands = []
                for task in self.tasks:
                    if k in task.budget_per_block:
                        demands.append(task.budget_per_block[k].epsilon(alpha))
                    else:
                        demands.append(0)
                m.addConstr(
                    sum([x[0, z] * demands[z] for z in range(n)])
                    - (1 - a[k, i]) * demands_upper_bound[(k, alpha)]
                    <= block.budget.epsilon(alpha)
                )
        print(m)

        # Objective function
        m.setObjective(x.sum(), GRB.MAXIMIZE)
        m.optimize()

        return [bool((abs(x[0, i].x - 1) < 1e-4)) for i in range(n)]


def main():
    # num_blocks = 1 # single-block case
    num_blocks = 2  # multi-block case

    blocks = {}
    for i in range(num_blocks):
        blocks[i] = Block.from_epsilon_delta(i, 10, 0.001)

    tasks = (
            [
                UniformTask(
                    id=i, profit=1, block_ids=range(num_blocks), budget=GaussianCurve(s)
                )
                for i, s in enumerate(np.linspace(0.1, 1, 10))
            ]
            + [
                UniformTask(
                    id=i,
                    profit=1,
                    block_ids=range(num_blocks),
                    budget=LaplaceCurve(l),
                )
                for i, l in enumerate(np.linspace(0.1, 10, 5))
            ]
            + [
                UniformTask(
                    id=i,
                    profit=1,
                    block_ids=range(num_blocks),
                    budget=SubsampledGaussianCurve.from_training_parameters(
                        60_000, 64, 10, s
                    ),
                )
                for i, s in enumerate(np.linspace(1, 10, 5))
            ]
    )

    random.shuffle(tasks)
    scheduler = Simplex(tasks, blocks)
    allocation = scheduler.schedule()
    print(allocation)
    # self.config.plotter.plot(tasks, blocks, allocation)


if __name__ == "__main__":
    main()
