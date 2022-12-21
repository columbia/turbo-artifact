from privacypacking.planner.planner import Planner
from privacypacking.cache.cache import A, R
import gurobipy as gp
from gurobipy import GRB
import time
import math
import os
from collections import namedtuple
from sortedcollections import OrderedSet
import random
from multiprocessing import Manager, Process
from privacypacking.utils.compute_utility_curve import compute_utility_curve
from privacypacking.budget.utils import from_pure_epsilon_to_budget

from termcolor import colored

Chunk = namedtuple("Chunk", ["index", "total_pure_epsilon"])


class ILP(Planner):
    def __init__(self, cache, blocks, utility, objective, variance_reduction):
        super().__init__(cache)
        self.blocks = blocks
        self.utility = utility
        self.p = 0.00001            # Probability that accuracy won't be respected
        self.max_pure_epsilon = 0.5
        self.sequencial = False
        self.objective = objective
        self.variance_reduction = variance_reduction
        self.C = {}
        self.B = {}

    def get_execution_plan(self, query_id, block_request, _):
        n = len(block_request)
        offset = block_request[0]

        block_budgets = self.get_block_budgets(block_request)
        block_request = (block_request[0], block_request[-1])
        indices = self.get_chunk_indices(n, offset)
        self.get_chunk_pure_epsilon_cost(query_id, indices, offset, self.C)
        self.get_chunk_available_budget(
            indices, block_budgets, self.B
        )  # minimum available pure_epsilon across the blocks
        f = compute_utility_curve(self.utility, self.p, n, self.max_pure_epsilon)
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
                            self.C,
                            self.B,
                            block_budgets,
                            f,
                            n,
                            indices,
                            self.objective,
                            self.variance_reduction,
                            self.max_pure_epsilon,
                        ),
                    )
                )
                processes[i].start()

            for i in range(num_processes):
                processes[i].join()

        # Find the best solution
        plan = None
        best_objvalue1 = best_objvalue2 = math.inf

        for k, (chunks, objvalue1, objvalue2) in return_dict.items():
            if objvalue1 < best_objvalue1 or (
                (objvalue1 == best_objvalue1) and objvalue2 < best_objvalue2
            ):
                plan = A(
                    [
                        R(query_id, (i + offset, j + offset), from_pure_epsilon_to_budget(pure_epsilon))
                        for ((i, j), pure_epsilon) in chunks
                    ]
                )
                best_objvalue1 = objvalue1
                best_objvalue2 = objvalue2

        if plan is not None:
            print(
                colored(
                    f"Got plan (cost={best_objvalue1}, {best_objvalue2}) for blocks {block_request}: {plan}",
                    "yellow",
                )
            )
        return plan

    def get_block_budgets(self, block_request):
        return [self.blocks[block_id].budget.epsilon for block_id in block_request]

    def get_chunk_pure_epsilon_cost(self, query_id, indices, offset, C):
        
        def get_cache_entry_budget(query_id, blocks):
            result, budget, _ = self.cache.get_entry(query_id, blocks)
            if result is not None:
                return budget.pure_epsilon
            return 0.0

        for (i, j) in indices:
            C[(i, j)] = get_cache_entry_budget(query_id, (i + offset, j + offset))

    def get_chunk_available_budget(self, indices, block_budgets, B):
        for (i, j) in indices:
            B[(i, j)] = min(block_budgets[i : j + 1])

    def get_chunk_indices(self, n, offset):
        indices = OrderedSet()
        for i in range(n):
            for j in range(i, n):
                if self.satisfies_constraint((i + offset, j + offset)):
                    indices.add(((i, j)))
        return indices

    def satisfies_constraint(self, blocks, branching_factor=2):
        size = blocks[1] - blocks[0] + 1
        if not math.log(size, branching_factor).is_integer():
            return False
        if (blocks[0] % size) != 0:
            return False
        return True

def solve(
    kmin,
    kmax,
    return_dict,
    C,
    B,
    block_budgets,
    f,
    n,
    indices,
    objective_function,
    variance_reduction,
    max_pure_epsilon,
):
    t = time.time()
    for k in range(kmin, kmax + 1):
        chunks, objval1, objval2 = solve_gurobi(
            f[k],
            max_pure_epsilon,
            k,
            n,
            indices,
            C,
            B,
            block_budgets,
            variance_reduction,
            objective_function,
        )
        if chunks:
            return_dict[k] = (chunks, objval1, objval2)

    t = (time.time() - t) / (kmax + 1 - kmin)
    # print(f"    Took:  {t}")


def solve_gurobi(
    min_pure_epsilon,
    max_pure_epsilon,
    K,
    N,
    indices,
    C,
    B,
    block_budgets,
    variance_reduction,
    objective_function,
):
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)
        m.Params.OutputFlag = 0

        new_pure_epsilon_per_chunk = {}
        total_pure_epsilon_per_chunk = {}

        # A variable per chunk
        x = m.addVars(
            [(i, j) for (i, j) in indices],
            vtype=GRB.INTEGER,
            lb=0,
            ub=1,
            name="x",
        )

        # "total_pure_epsilon": total pure_epsilon that can be given to each chunk either because
        # it can be consumed from the blocks (not exceeding threshold) or because we have it in cache
        # "new_pure_epsilon": the new pure_epsilon that must be spent in order to reach "total_pure_epsilon"

        if objective_function == "minimize_budget":

            for (i, j) in indices:

                new_pure_epsilon = max(min_pure_epsilon - C[(i, j)], 0)
                if (
                    not variance_reduction and new_pure_epsilon > 0
                ):  # If the result was not stored with enough budget
                    new_pure_epsilon = (
                        min_pure_epsilon  # Turning off variance reduction
                    )

                total_pure_epsilon = max(min_pure_epsilon, C[(i, j)])

                # Enough budget in blocks constraint to accommodate "new_pure_epsilon"
                # If at least one block in chunk i,j cannot give minimum budget 'new_pure_epsilon' then Xij=0
                for k in range(N):  # For every block
                    if from_pure_epsilon_to_budget(new_pure_epsilon) > block_budgets[k]:
                        m.addConstr(x[i, j] == 0)
                        break

                new_pure_epsilon_per_chunk[(i, j)] = new_pure_epsilon * (j - i + 1)
                total_pure_epsilon_per_chunk[
                    (i, j)
                ] = (
                    -total_pure_epsilon
                )  # multiplying by -1: we want to maximize the total total_pure_epsilon so we want to minimize the total -1*total_pure_epsilon

        elif objective_function == "minimize_error":

            for (i, j) in indices:

                total_pure_epsilon = max(
                    min(B[(i, j)], max_pure_epsilon), C[(i, j)]
                )  # B[(i,j)]: maximum budget blocks in chunk (i,j) can give

                if (
                    total_pure_epsilon < min_pure_epsilon
                ):  # If the budget that can be provided to the chunk is less than the budget demanded to maintain accuracy we invalidate the chunk
                    m.addConstr(x[i, j] == 0)

                new_pure_epsilon = (
                    total_pure_epsilon - C[(i, j)]
                )  # new_pure_epsilon between 0 and max_pure_epsilon
                if not variance_reduction and new_pure_epsilon > 0:
                    new_pure_epsilon = total_pure_epsilon

                new_pure_epsilon_per_chunk[(i, j)] = new_pure_epsilon * (j - i + 1)
                total_pure_epsilon_per_chunk[
                    (i, j)
                ] = (
                    -total_pure_epsilon
                )  # multiplying by -1: we want to maximize the total total_pure_epsilon so we want to minimize the total -1*total_pure_epsilon

                # print("new_pure_epsilon", (i,j), total_pure_epsilon_per_chunk[(i, j)])
                # print("total_pure_epsilon", (i,j), total_pure_epsilon_per_chunk[(i, j)])

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

        # Aggregations must be at most K-1 constraint
        m.addConstr((gp.quicksum(x[i, j] for (i, j) in indices)) <= K)

        # Objective function
        if objective_function == "minimize_budget":
            m.setObjectiveN(x.prod(new_pure_epsilon_per_chunk), 0, 1)  # Primary objective
            m.setObjectiveN(x.prod(total_pure_epsilon_per_chunk), 1, 0)  # Secondary objective
        elif objective_function == "minimize_error":
            m.setObjectiveN(x.prod(total_pure_epsilon_per_chunk), 0, 1)
            m.setObjectiveN(x.prod(new_pure_epsilon_per_chunk), 1, 0)
        # elif objective ==  "minimize_aggregations":
        # m.setObjectiveN(x.sum(), 0, 1)
        # m.setObjectiveN(x.prod(budget_cost), 1, 0)

        m.ModelSense = GRB.MINIMIZE
        m.optimize()

        if m.status == GRB.OPTIMAL:
            m.params.ObjNumber = 0
            objval1 = m.ObjNVal
            m.params.ObjNumber = 1
            objval2 = m.ObjNVal

            if objective_function == "minimize_budget":
                objval2 = -math.pow(K, 1.5) / objval2
            if objective_function == "minimize_error":
                objval1 = -math.pow(K, 1.5) / objval1

            chunks = []
            for (i, j) in indices:
                if int((abs(x[i, j].x - 1) < 1e-4)) == 1:
                    chunks.append(Chunk((i, j), -total_pure_epsilon_per_chunk[(i, j)]))
            return chunks, objval1, objval2
        return [], math.inf, math.inf


def test():
    def run(request):
        dplanner = ILP(None, None, None, 100)
        dplanner.get_execution_plan(None, request, None)

    request = list(range(10))
    run(request)


if __name__ == "__main__":
    test()
