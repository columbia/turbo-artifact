from privacypacking.planner.planner import Planner
from privacypacking.cache.cache import A, R
from privacypacking.budget import BasicBudget
import gurobipy as gp
from gurobipy import GRB
import time
import math
import os
import random
from multiprocessing import Manager, Process
from termcolor import colored


class ILP(Planner):
    def __init__(self, cache, blocks, utility, objective, variance_reduction):
        super().__init__(cache)
        self.blocks = blocks
        self.delta = 0.00001
        self.utility = utility
        self.emax = 0.7
        self.sequencial = False
        self.objective = objective
        self.variance_reduction = variance_reduction

    def get_execution_plan(self, query_id, block_request, _):
        n = len(block_request)
        offset = block_request[0]

        block_budgets = self.get_block_budgets(block_request)
        block_request = (block_request[0], block_request[-1])
        indices = self.get_indices(n, offset)
        C = self.get_costs(query_id, indices, offset)

        # Collect all valid aggregations
        f = {}
        for k in range(1, n + 1):
            e = self.f(k, self.delta, self.utility)
            if e > self.emax:
                break
            f[k] = e
        max_k = len(f)

        if max_k == 0:
            return None

        if not self.sequencial:  # Running in Parallel
            processes = []
            manager = Manager()
            return_dict = manager.dict()
            num_processes = min(os.cpu_count(), max_k)

            k = max_k // num_processes
            for i in range(num_processes):
                k_start = i * k + 1
                k_end = i * k + k if i * k + k < max_k else max_k
                processes.append(
                    Process(
                        target=solve,
                        args=(
                            k_start,
                            k_end,
                            return_dict,
                            C,
                            block_budgets,
                            f,
                            n,
                            indices,
                            self.objective,
                            self.variance_reduction,
                        ),
                    )
                )
                processes[i].start()

            for i in range(num_processes):
                processes[i].join()
        else:  # Running sequentially
            return_dict = dict()
            solve(
                1, max_k, return_dict, C, block_budgets, f, n, indices, self.objective, self.variance_reduction,
            )

        # Find the best solution
        best_solution = best_budget = None
        best_objvalue1 = math.inf
        best_objvalue2 = math.inf
    
        for k, (solution, objvalue1, objvalue2, budget) in return_dict.items():
            if objvalue1 < best_objvalue1 or ((objvalue1 == best_objvalue1) and objvalue2 < best_objvalue2):
                best_solution, best_objvalue1, best_objvalue2, best_budget = solution, objvalue1, objvalue2, budget
            # if objvalue == 0:
                # break

        if not math.isinf(best_objvalue1):
            plan = []
            for (i, j) in best_solution:
                plan += [R(query_id, (i + offset, j + offset), best_budget)]
            plan = A(plan, budget=best_budget)
            print(
                colored(
                    f"Got plan (cost={best_objvalue1}, {best_objvalue2}) for blocks {block_request}: {plan}, {plan.budget}",
                    "yellow",
                )
            )
            return plan
        return None

    def get_block_budgets(self, block_request):
        # return [5 for _ in block_request]
        return [
            self.blocks[block_id].budget.epsilon for block_id in block_request
        ]  # available budget of blocks

    def get_costs(self, query_id, indices, offset):
        C = {}
        for (i, j) in indices:
            C[(i, j)] = self.cache.get_entry_budget(query_id, (i+offset, j+offset))
            # print((i+offset, j+offset), C[(i, j)])
            # C[(i,j)] = random.uniform(0, 5)
        return C

    def get_indices(self, n, offset):
        indices = set()
        for i in range(n):
            for j in range(i, n):
                if self.satisfies_constraint((i+offset, j+offset)):
                    indices.add(((i, j)))
        return indices

    def satisfies_constraint(self, blocks, branching_factor=2):
        size = blocks[1] - blocks[0] + 1
        if not math.log(size, branching_factor).is_integer():
            return False
        if (blocks[0] % size) != 0:
            return False
        return True

    def f(self, k, delta, u):
        return (math.sqrt(8 * k) * math.log(2 / delta)) / u


def solve(kmin, kmax, return_dict, C, block_budgets, f, n, indices, objective, variance_reduction):
    t = time.time()
    for k in range(kmin, kmax + 1):
        budget = BasicBudget(f[k])
        solution, objval1, objval2 = solve_gurobi(
            budget.epsilon, k, n, indices, C, block_budgets, variance_reduction, objective
        )
        if solution:
            return_dict[k] = (solution, objval1, objval2, budget)

    t = (time.time() - t) / (kmax + 1 - kmin)
    # logger.warning(f"Optimization took: {time.time() - t}")
    # print(f"    Took:  {t}")


def solve_gurobi(budget_demand, K, N, indices, C, block_budgets, variance_reduction, objective):
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)

        m.Params.OutputFlag = 0
        # m.Params.TimeLimit = self.simulator_config.metric.gurobi_timeout
        # self.simulator_config.metric.gurobi_threads
        # m.Params.LogToConsole = 0

        # m.Params.MIPGap = 0.01  # Optimize within 1% of optimal
        # m = gp.Model("pack")

        cost_per_block_per_chunk = {}
        for (i, j) in indices:
            cost_per_block = max(budget_demand - C[(i, j)], 0)
            if (not variance_reduction and cost_per_block > 0):  # If the result was not stored with enough budget
                cost_per_block = budget_demand  # Turning off optimization
            cost_per_block_per_chunk[(i, j)] = cost_per_block

        # if objective == "minimize_budget":
        coeffs = {}
        for (i, j) in indices:
            num_blocks = j - i + 1
            coeffs[(i, j)] = cost_per_block_per_chunk[(i, j)] * num_blocks

        # A variable per chunk
        x = m.addVars(
            [(i, j) for (i, j) in indices],
            vtype=GRB.INTEGER,
            lb=0,
            ub=1,
            name="x",
        )

        # No overlapping chunks constraint
        for k in range(N):  # For every block
            m.addConstr(
                (
                    gp.quicksum(
                        x[i, j]
                        for i in range(k + 1)
                        for j in range(k, N)
                        if (i, j) in indices
                    )
                )
                == 1
            )

        # Enough budget in blocks constraint
        for k in range(N):  # For every block
            for i in range(k + 1):
                for j in range(k, N):
                    if (i, j) in indices:
                        m.addConstr(
                            x[i, j] * cost_per_block_per_chunk[(i, j)]
                            <= block_budgets[k]
                        )

        # Aggregations must be at most K-1 constraint
        m.addConstr((gp.quicksum(x[i, j] for (i, j) in indices)) <= K)

        # Objective function
        if objective == "minimize_budget":
            m.setObjectiveN(x.prod(coeffs), 0, 1)
            m.setObjectiveN(x.sum(), 1, 0)
            # m.setObjective(x.prod(coeffs), GRB.MINIMIZE)
        elif objective ==  "minimize_aggregations":
            m.setObjectiveN(x.sum(), 0, 1)
            m.setObjectiveN(x.prod(coeffs), 1, 0)
            # m.setObjective(x.sum(), GRB.MINIMIZE)

        m.ModelSense = GRB.MINIMIZE
        m.optimize()
        # logger.warning(f"status: {m.Status} timeout? {m.Status == GRB.TIME_LIMIT}")

        if m.Status == GRB.TIME_LIMIT:
            raise Exception(
                # f"Solver timeout after {self.simulator_config.metric.gurobi_timeout}s"
            )
        if m.status == GRB.OPTIMAL:
            m.params.ObjNumber = 0
            objval1 = m.ObjNVal
            m.params.ObjNumber = 1
            objval2 = m.ObjNVal
            return [
                (i, j) for (i, j) in indices if int((abs(x[i, j].x - 1) < 1e-4)) == 1
            ], objval1, objval2
        return [], math.inf, math.inf

def test():
    def run(request):
        dplanner = ILP(None, None, None, 100)
        dplanner.get_execution_plan(None, request, None)

    request = list(range(10))
    run(request)


if __name__ == "__main__":
    test()
