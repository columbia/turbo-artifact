import random

import gurobipy as gp
from gurobipy import GRB

from privacypacking.budget import ALPHAS, Block
from privacypacking.offline.schedulers.scheduler import Scheduler


class Simplex(Scheduler):
    def __init__(self, tasks, blocks):
        super().__init__(tasks, blocks)

    def schedule(self):

        """
        Returns a list of booleans corresponding to the tasks that are allocated
        """
        m = gp.Model("pack")
        n = len(self.tasks)
        d = len(ALPHAS)

        demands_upper_bound = {}
        for k, block in enumerate(self.blocks):
            for alpha in block.budget.alphas:
                demands_upper_bound[(k, alpha)] = 0
                for task in self.tasks:
                    if k in task.budget_per_block:
                        demands_upper_bound[(k, alpha)] += task.budget_per_block[
                            k
                        ].epsilon(alpha)

                # Only non-zero blocks are tracked
                # sum([task.budget_per_block[k].epsilon(alpha) for task in self.tasks])

        # Variables
        x = m.addMVar((1, n), vtype=GRB.BINARY, name="x")
        a = m.addMVar((len(self.blocks), d), vtype=GRB.BINARY, name="a")
        print(x, a)

        # Constraints
        for k, _ in enumerate(self.blocks):
            m.addConstr(a[k].sum() >= 1)

        for k, block in enumerate(self.blocks):
            for i, alpha in enumerate(block.budget.alphas):
                demands = []
                for task in self.tasks:
                    if k in task.budget_per_block:
                        demands.append(task.budget_per_block[k].epsilon(alpha))
                    else:
                        demands.append(0)
                # demands = [
                #     task.budget_per_block[k].epsilon(alpha)
                #     for j, task in enumerate(self.tasks)
                # ]
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

    blocks = [Block.from_epsilon_delta(i, 10, 0.001) for i in range(num_blocks)]
    tasks = (
        [
            create_gaussian_task(i, num_blocks, range(num_blocks), s)
            for i, s in enumerate(np.linspace(0.1, 1, 10))
        ]
        + [
            create_laplace_task(i, num_blocks, range(num_blocks), l)
            for i, l in enumerate(np.linspace(0.1, 10, 5))
        ]
        + [
            create_subsamplegaussian_task(
                i, num_blocks, range(num_blocks), ds=60_000, bs=64, epochs=10, s=s
            )
            for i, s in enumerate(np.linspace(1, 10, 5))
        ]
    )

    random.shuffle(tasks)
    scheduler = Simplex(tasks, blocks)
    allocation = scheduler.schedule()
    scheduler.plot(allocation)


if __name__ == "__main__":
    main()
