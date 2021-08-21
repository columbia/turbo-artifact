import random
from typing import List

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
from privacypacking.schedulers.scheduler import Scheduler


class Simplex(Scheduler):
    def __init__(self, tasks, blocks, config=None):
        super().__init__(tasks, blocks, config)

    def solve_allocation(self) -> List[bool]:

        """
        Returns a list of booleans corresponding to the tasks that are allocated
        """
        m = gp.Model("pack")

        # TODO: alphas from which block? Which subset?
        alphas = ALPHAS
        task_ids = [t.id for t in self.tasks]
        block_ids = [k for k in self.blocks]

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
        x = m.addVars(task_ids, vtype=GRB.BINARY, name="x")
        a = m.addVars(
            [(k, alpha) for alpha in alphas for k in block_ids],
            vtype=GRB.BINARY,
            name="a",
        )

        # Constraints
        for k, _ in enumerate(self.blocks):
            m.addConstr(a.sum(k, "*") >= 1)

        for k, block in self.blocks.items():
            for alpha in block.budget.alphas:
                demands_k_alpha = {
                    t.id: t.get_budget(k).epsilon(alpha) for t in self.tasks
                }
                m.addConstr(
                    x.prod(demands_k_alpha)
                    - (1 - a[k, alpha]) * demands_upper_bound[(k, alpha)]
                    <= block.budget.epsilon(alpha)
                )

        # Objective function
        profits = {task.id: task.profit for task in self.tasks}
        m.setObjective(x.prod(profits), GRB.MAXIMIZE)
        m.optimize()

        return [bool((abs(x[i].x - 1) < 1e-4)) for i in task_ids]

    def schedule(self) -> List[int]:
        allocated_task_ids = []
        allocation = self.solve_allocation()
        for i, allocated in enumerate(allocation):
            if allocated:
                allocated_task_ids.append(self.tasks[i].id)
                self.consume_budgets(self.tasks[i])
        return allocated_task_ids