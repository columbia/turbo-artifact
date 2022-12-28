import math
import os
import time
from collections import namedtuple
from multiprocessing import Manager, Process

import gurobipy as gp
from gurobipy import GRB
from sortedcollections import OrderedSet
from termcolor import colored
import numpy as np
from privacypacking.cache.cache import A, R
from privacypacking.planner.planner import Planner
from privacypacking.utils.compute_utility_curve import compute_utility_curve
from privacypacking.budget.curves import LaplaceCurve


Chunk = namedtuple("Chunk", ["index", "noise_std"])


class ILP(Planner):
    def __init__(self, cache, blocks, planner_args):
        assert planner_args.enable_caching == True
        assert planner_args.enable_dp == True
        super().__init__(cache, blocks, planner_args)
        self.sequencial = False
        self.C = {}

    def get_execution_plan(self, query_id, utility, utility_beta, block_request):
        n = len(block_request)
        offset = block_request[0]

        block_budgets = self.get_block_budgets(block_request)
        block_request = (block_request[0], block_request[-1])
        indices = self.get_chunk_indices(n, offset)
        self.get_chunk_cached_noise_std(query_id, indices, offset, self.C)

        f = {}
        for k in range(1, n + 1):
            f[k] = compute_utility_curve(utility, utility_beta, k)
            if self.max_pure_epsilon is not None and f[k] > self.max_pure_epsilon:
                break  # No more aggregations allowed
        max_k = len(f)

        if max_k == 0:  # User-accuracy too high - max pure-epsilon always exceeded
            return None

        # Running in Parallel
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
                        block_budgets,
                        f,
                        n,
                        indices,
                        self.variance_reduction,
                    ),
                )
            )
            processes[i].start()

        for i in range(num_processes):
            processes[i].join()

        # Find the best solution
        plan = None
        best_objvalue = math.inf

        for k, (chunks, objvalue) in return_dict.items():
            if objvalue < best_objvalue:
                plan = A(
                    query_id,
                    [
                        R((i + offset, j + offset), noise_std)
                        for ((i, j), noise_std) in chunks
                    ],
                )
                best_objvalue = objvalue

        if plan is not None:
            print(
                colored(
                    f"Got plan (cost={best_objvalue}) for blocks {block_request}: {plan}",
                    "yellow",
                )
            )
        return plan

    def get_block_budgets(self, block_request):
        return [self.blocks[block_id].budget.epsilon for block_id in block_request]

    def get_chunk_cached_noise_std(self, query_id, indices, offset, C):
        for (i, j) in indices:
            cache_entry = self.cache.get_entry(query_id, (i + offset, j + offset))
            if cache_entry is not None:
                C[(i, j)] = cache_entry.noise_std

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


def solve(kmin, kmax, return_dict, C, block_budgets, f, n, indices, variance_reduction):
    t = time.time()
    for k in range(kmin, kmax + 1):
        laplace_scale = (
            1 / f[k]
        )  # f(k): Minimum pure epsilon for reaching accuracy target given k
        target_noise_std = math.sqrt(2) * laplace_scale
        chunks, objval = solve_gurobi(
            target_noise_std, k, n, indices, C, block_budgets, variance_reduction
        )
        if chunks:
            return_dict[k] = (chunks, objval)
    t = (time.time() - t) / (kmax + 1 - kmin)
    # print(f"    Took:  {t}")


def solve_gurobi(target_noise_std, K, N, indices, C, block_budgets, vr):
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)
        m.Params.OutputFlag = 0

        # An indicator of how much budget we'll spend
        # We want to minimize this
        fresh_pure_epsilon_per_chunk = {}

        # A variable per chunk
        x = m.addVars(
            [(i, j) for (i, j) in indices],
            vtype=GRB.INTEGER,
            lb=0,
            ub=1,
            name="x",
        )

        for (i, j) in indices:
            if (i, j) in C and target_noise_std >= C[
                (i, j)
            ].noise_std:  # Good enough estimate in the cache
                fresh_pure_epsilon = 0
            else:  # Need to improve on the cache
                # TODO: re-enable variance reduction
                laplace_scale = target_noise_std / np.sqrt(2)
                fresh_pure_epsilon = 1 / laplace_scale

                # Enough budget in blocks constraint to accommodate "new_pure_epsilon"
                # If at least one block in chunk i,j does not have enough budget then Xij=0
                run_budget = LaplaceCurve(laplace_noise=laplace_scale)
                for k in range(N):  # For every block
                    if run_budget > block_budgets[k]:
                        m.addConstr(x[i, j] == 0)
                        break

            fresh_pure_epsilon_per_chunk[(i, j)] = fresh_pure_epsilon * (j - i + 1)

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
        m.setObjective(x.prod(fresh_pure_epsilon_per_chunk))
        m.ModelSense = GRB.MINIMIZE
        m.optimize()

        if m.status == GRB.OPTIMAL:
            chunks = []
            for (i, j) in indices:
                if int((abs(x[i, j].x - 1) < 1e-4)) == 1:
                    chunks.append(Chunk((i, j), target_noise_std))
            return chunks, m.ObjVal
        return [], math.inf
