import os
import math
import time
import numpy as np
from loguru import logger
from termcolor import colored
from collections import namedtuple
from sortedcollections import OrderedSet
from multiprocessing import Manager, Process

import gurobipy as gp
from gurobipy import GRB

from executor import A, R
from planner.planner import Planner
from budget.curves import ZeroCurve
from utils.compute_utility_curve import compute_utility_curve


Chunk = namedtuple("Chunk", ["index", "noise_std"])


class ILP(Planner):
    def __init__(self, cache, blocks, planner_args):
        assert planner_args.get("enable_caching") == True
        assert planner_args.get("enable_dp") == True
        super().__init__(cache, blocks, **planner_args)

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

    def get_execution_plan(self, query_id, utility, utility_beta, block_request):
        n = len(block_request)
        offset = block_request[0]

        requested_blocks = [self.blocks[block_id] for block_id in block_request]
        block_request = (block_request[0], block_request[-1])
        indices = self.get_chunk_indices(n, offset)

        #######################################################################
        f = {}
        for k in range(1, n + 1):
            f[k] = compute_utility_curve(utility, utility_beta, k)
            if self.max_pure_epsilon is not None and f[k] > self.max_pure_epsilon:
                break  # No more aggregations allowed
        max_k = len(f)
        # TODO: Delete this and find another way to obtain max_k
        #######################################################################

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
                        self.cache,
                        query_id,
                        requested_blocks,
                        n,
                        indices,
                        utility,
                        utility_beta,
                    ),
                )
            )
            processes[i].start()

        for i in range(num_processes):
            processes[i].join()

        plan = None
        best_objvalue = math.inf

        # Find the best solution
        for k, (chunks, objvalue) in return_dict.items():
            if objvalue < best_objvalue:
                plan = A(
                    query_id=query_id,
                    l=[
                        R((i + offset, j + offset), noise_std)
                        for ((i, j), noise_std) in chunks
                    ],
                    cost=objvalue,
                )
                best_objvalue = objvalue
        return plan


def solve(
    kmin,
    kmax,
    return_dict,
    cache,
    query_id,
    requested_blocks,
    n,
    indices,
    utility,
    utility_beta,
):
    t = time.time()
    for k in range(kmin, kmax + 1):
        chunks, objval = optimize(
            k,
            n,
            indices,
            cache,
            query_id,
            requested_blocks,
            utility,
            utility_beta,
        )
        if chunks:
            return_dict[k] = (chunks, objval)
    t = (time.time() - t) / (kmax + 1 - kmin)
    # print(f"    Took:  {t}")


def optimize(
    K,
    N,
    indices,
    cache,
    query_id,
    requested_blocks,
    utility,
    utility_beta,
):
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)
        m.Params.OutputFlag = 0

        # An indicator of how much budget we'll spend - we want to minimize this
        fresh_pure_epsilon_per_chunk = {}

        # A variable per chunk
        x = m.addVars(
            [(i, j) for (i, j) in indices],
            vtype=GRB.INTEGER,
            lb=0,
            ub=1,
            name="x",
        )

        # A variable for the cache type
        # y = m.addVar(
        # 0.0, 1.0, 1.0, GRB.BINARY, "y"
        # )

        for (i, j) in indices:
            # dont look at this
            # f(k,i,j) = minimum budget that must be used for this chunk
            # fresh_pure_epsilon_det = max(f(k,i,j)-cache.estimate_run_budget(query_id, hyperblock, noise_std), 0)
            # fresh_pure_epsilon_prob = f(k,i,j)    probabilistic?
            # fresh_pure_epsilon = y*fresh_pure_epsilon_det - (1-y)*fresh_pure_epsilon_prob

            # hyperblock = HyperBlock(
            #     {
            #         requested_blocks[idx].id: requested_blocks[idx]
            #         for idx in range(i, j + 1)
            #     }
            # )
            laplace_scale = 1 / compute_utility_curve(
                utility, utility_beta, K
            )  # This will also depend on i and j and will be different for each chunk
            target_noise_std = math.sqrt(2) * laplace_scale
            run_budget = cache.estimate_run_budget(
                query_id, hyperblock, target_noise_std
            )
            # still working with pure epsilon here instead of std
            fresh_pure_epsilon = (
                0 if isinstance(run_budget, ZeroCurve) else run_budget.pure_epsilon
            )





            # Enough budget in blocks constraint to accommodate "fresh_pure_epsilon"
            # If at least one block in chunk i,j does not have enough budget then Xij=0
            for k in range(i, j + 1):  # For every block in the chunk i,j
                if not (requested_blocks[k].budget - run_budget).is_positive():
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
