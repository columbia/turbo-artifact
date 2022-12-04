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
    def __init__(self, cache, blocks, utility):
        super().__init__(cache)
        self.blocks = blocks
        self.delta = 0.00001
        self.utility = utility
        self.emax = 0.5
        self.sequencial = False

    def get_execution_plan(self, query_id, block_request, _):
        n = len(block_request)
        block_start = block_request[0]

        block_budgets = self.get_block_budgets(block_request)
        block_request = (block_request[0], block_request[-1])
        indices = self.get_indices(n)
        C = self.get_costs(query_id, indices)

        # Collect all valid aggregations
        f = {}
        for k in range(1, n + 1):
            e = self.f(k, self.delta, self.utility)
            if e > self.emax:
                break
            f[k] = e
        max_k = len(f)

        if not self.sequencial:     # Running in Parallel - (hand tuned)
            processes = []
            manager = Manager()
            return_dict = manager.dict()
            num_processes = min(os.cpu_count(), max_k)

            k = max_k // num_processes
            i=-1
            for i in range(num_processes - 1):
                processes.append(
                    Process(
                        target=solve,
                        args=(i*k+1,  i*k+k, return_dict, C, block_budgets, f,  n, indices),
                    )
                )
                processes[i].start()
            i += 1
            processes.append(
                Process(
                    target=solve, args=(i*k+1, max_k, return_dict, C, block_budgets, f,  n, indices)
                )
            )
            processes[i].start()

            for i in range(num_processes):
                processes[i].join()
        else:           # Running sequentially
            return_dict = dict()
            solve(1, max_k, return_dict, C, block_budgets, f,  n, indices)

        # Find the best solution
        best_solution = best_budget = None
        best_objvalue = math.inf
        
        for k, (solution, objvalue, budget) in return_dict.items():
            if objvalue <= best_objvalue:
                best_solution = solution
                best_objvalue = objvalue
                best_budget = budget

            if objvalue == 0:
                break

        if not math.isinf(best_objvalue):
            plan = []
            for (i, j) in best_solution:
                plan += [R(query_id, (i + block_start, j + block_start), best_budget)]
            plan = A(plan, budget=best_budget)
            print(
                colored(
                    f"Got plan (cost={best_objvalue}) for blocks {block_request}: {plan}, {plan.budget}",
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

    def get_costs(self, query_id, indices):
        C = {}
        for (i, j) in indices:
            C[(i, j)] = self.cache.get_entry_budget(query_id, (i, j))
            # C[(i,j)] = random.uniform(0, 5)
        return C

    def get_indices(self, n):
        indices = set()
        for i in range(n):
            for j in range(i, n):
                if self.satisfies_constraint((i, j)):
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
        

def solve(kmin, kmax, return_dict, C, block_budgets, f, n, indices):
    for k in range(kmin, kmax+1):
        budget = BasicBudget(f[k])
        solution, objval = solve_gurobi(budget.epsilon, k, n, indices, C, block_budgets)
        if solution:
            return_dict[k] = (solution, objval, budget)

def solve_gurobi(budget_demand, K, N, indices, C, block_budgets):
    t = time.time()

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

        # Set coefficients
        coeffs = {}
        cost_per_block_per_chunk = {}
        for (i, j) in indices:
            num_blocks = j - i + 1
            cost_per_block = max(budget_demand - C[(i, j)], 0)
            if (
                cost_per_block > 0
            ):  # If the result was not stored with enough budget
                cost_per_block = budget_demand  # Turning off optimization for now
            cost_per_block_per_chunk[(i, j)] = cost_per_block
            coeffs[(i, j)] = cost_per_block * num_blocks

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
        m.setObjective(x.prod(coeffs), GRB.MINIMIZE)
        m.optimize()

        print(f"    K {K}, e {budget_demand} took:  {time.time() - t}")
        # logger.warning(f"Optimization took: {time.time() - t}")
        # logger.warning(f"status: {m.Status} timeout? {m.Status == GRB.TIME_LIMIT}")

        if m.Status == GRB.TIME_LIMIT:
            raise Exception(
                # f"Solver timeout after {self.simulator_config.metric.gurobi_timeout}s"
            )
        if m.status == GRB.OPTIMAL:
            return [
                (i, j)
                for (i, j) in indices
                if int((abs(x[i, j].x - 1) < 1e-4)) == 1
            ], m.objval
        return [], math.inf


def test():
    def run(request):
        dplanner = ILP(None, None, None, 100)
        dplanner.get_execution_plan(None, request, None)

    request = list(range(10))
    run(request)


if __name__ == "__main__":
    test()
