import time
from typing import List

import gurobipy as gp
from gurobipy import GRB
from loguru import logger
from mip import BINARY, Model, maximize, xsum

from privacypacking.budget import ALPHAS
from privacypacking.schedulers.scheduler import Scheduler
from privacypacking.utils.utils import GUROBI, MIP


class Simplex(Scheduler):
    def __init__(self, solver=MIP):
        super().__init__()
        self.solver = solver

    def solve_allocation_cbc(self, tasks) -> List[bool]:
        m = Model("pack")
        alphas = ALPHAS
        task_ids = [t.id for t in tasks]
        block_ids = [k for k in self.blocks]
        demands_upper_bound = {}
        for k, block in self.blocks.items():
            for alpha in block.budget.alphas:
                demands_upper_bound[(k, alpha)] = 0
                for task in tasks:
                    if k in task.budget_per_block:
                        demands_upper_bound[(k, alpha)] += task.budget_per_block[
                            k
                        ].epsilon(alpha)

        # Variables
        x = [m.add_var(var_type=BINARY) for i in task_ids]
        a = [{alpha: m.add_var(var_type=BINARY) for alpha in alphas} for k in block_ids]

        # Constraints
        for k, _ in enumerate(self.blocks):
            m += xsum(a[k][alpha] for alpha in alphas) >= 1

        for k, block in self.blocks.items():
            for alpha in block.budget.alphas:
                prod = xsum(
                    t.get_budget(k).epsilon(alpha) * x[i] for i, t in enumerate(tasks)
                )
                m += prod - (1 - a[k][alpha]) * demands_upper_bound[
                    (k, alpha)
                ] <= block.budget.epsilon(alpha)

        # Objective function
        m.objective = maximize(xsum(t.profit * x[i] for i, t in enumerate(tasks)))
        m.optimize()

        return [bool((abs(x[i].x - 1) < 1e-4)) for i in task_ids]

    def solve_allocation_gurobi(self, tasks) -> List[bool]:

        """
        Returns a list of booleans corresponding to the tasks that are allocated
        """

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            m = gp.Model(env=env)

            # m.Params.OutputFlag = 0
            m.Params.TimeLimit = self.config.gurobi_timeout
            # m.Params.MIPGap = 0.01  # Optimize within 1% of optimal

            # m = gp.Model("pack")

            # TODO: alphas from which block? Which subset?
            alphas = ALPHAS
            task_ids = [t.id for t in tasks]
            block_ids = [k for k in self.blocks]

            demands_upper_bound = {}
            for k, block in self.blocks.items():
                for alpha in block.budget.alphas:
                    demands_upper_bound[(k, alpha)] = 0
                    for task in tasks:
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
                        t.id: t.get_budget(k).epsilon(alpha) for t in tasks
                    }
                    m.addConstr(
                        x.prod(demands_k_alpha)
                        - (1 - a[k, alpha]) * demands_upper_bound[(k, alpha)]
                        <= block.budget.epsilon(alpha)
                    )

            # Objective function
            profits = {task.id: task.profit for task in tasks}
            m.setObjective(x.prod(profits), GRB.MAXIMIZE)
            t = time.time()
            m.optimize()
            # logger.warning(f"Optimization took: {time.time() - t}")
            # logger.warning(f"status: {m.Status} timeout? {m.Status == GRB.TIME_LIMIT}")

            if m.Status == GRB.TIME_LIMIT:
                raise Exception(f"Solver timeout after {self.config.gurobi_timeout}s")
            return [bool((abs(x[i].x - 1) < 1e-4)) for i in task_ids]

    def schedule_queue(self) -> List[int]:
        tasks = self.task_queue.tasks
        allocated_tasks = []
        if self.solver == MIP:
            allocation = self.solve_allocation_cbc(tasks)
        else:
            allocation = self.solve_allocation_gurobi(tasks)

        for i, allocated in enumerate(allocation):
            if allocated:
                allocated_tasks.append(tasks[i])
        for task in allocated_tasks:
            self.allocate_task(task)

        return [task.id for task in allocated_tasks]
