import os
import math
import time
from loguru import logger
from termcolor import colored
from collections import namedtuple
from sortedcollections import OrderedSet
from multiprocessing import Manager, Process

import gurobipy as gp
from gurobipy import GRB

from precycle.executor import A, RDet, RProb
from precycle.planner.planner import Planner

# from precycle.utils.utils import get_blocks_size
from precycle.utils.compute_utility_curve import (
    deterministic_compute_utility_curve,
    probabilistic_compute_utility_curve,
)


DeterministicChunk = namedtuple("DeterministicChunk", ["index", "noise_std"])
ProbabilisticChunk = namedtuple("ChunProbabilisticChunkk", ["index", "alpha_beta"])
ILPArgs = namedtuple(
    "ILPArgs", ["task", "K", "indices", "cache", "budget_accountant", "chunk_sizes"]
)


def get_chunk_indices(blocks):
    offset = blocks[0]
    n = blocks[1] - blocks[0] + 1
    indices = OrderedSet()
    for i in range(n):
        for j in range(i, n):
            if satisfies_constraint((i + offset, j + offset)):
                indices.add(((i, j)))
    return indices


def satisfies_constraint(blocks, branching_factor=2):
    n = blocks[1] - blocks[0] + 1
    if not math.log(n, branching_factor).is_integer():
        return False
    if (blocks[0] % n) != 0:
        return False
    return True


def get_chunk_sizes(blocks, blocks_metadata):
    # Assuming all blocks have equal size for now
    # if not change this implementation
    n = blocks[1] - blocks[0] + 1
    block_size = blocks_metadata["block_size"]
    chunk_sizes = {}
    for i in range(n):
        for j in range(i, n):
            chunk_sizes[(i, j)] = block_size * (j - i + 1)
    return chunk_sizes


class ILP(Planner):
    def __init__(self, cache, budget_accountant, config):
        super().__init__(cache, budget_accountant, config)

    def get_execution_plan(self, task):
        num_blocks = task.blocks[1] - task.blocks[0] + 1

        def min_cuts():
            indices = OrderedSet()
            indices.add(((0, num_blocks - 1)))
            block_size = self.blocks_metadata["block_size"]
            chunk_sizes = {(0, num_blocks - 1): block_size * num_blocks}
            return_dict = dict()

            solve(
                1,
                1,
                self.cache_type,
                return_dict,
                self.cache,
                task,
                self.budget_accountant,
                indices,
                chunk_sizes,
                self.probabilistic_cfg,
            )
            return return_dict

        def max_cuts():
            indices = OrderedSet()
            for i in range(num_blocks):
                indices.add(((i, i)))
            block_size = self.blocks_metadata["block_size"]
            chunk_sizes = {index: block_size for index in indices}
            return_dict = dict()

            solve(
                num_blocks,
                num_blocks,
                self.cache_type,
                return_dict,
                self.cache,
                task,
                self.budget_accountant,
                indices,
                chunk_sizes,
                self.probabilistic_cfg,
            )
            return return_dict

        def optimal_cuts():
            indices = get_chunk_indices(task.blocks)
            chunk_sizes = get_chunk_sizes(task.blocks, self.blocks_metadata)
            max_chunks = num_blocks

            # Running in Parallel
            processes = []
            manager = Manager()
            return_dict = manager.dict()
            num_processes = min(os.cpu_count(), max_chunks)

            k = max_chunks // num_processes
            for i in range(num_processes):
                k_start = i * k + 1
                k_end = i * k + k if i * k + k < max_chunks else max_chunks
                processes.append(
                    Process(
                        target=solve,
                        args=(
                            k_start,
                            k_end,
                            self.cache_type,
                            return_dict,
                            self.cache,
                            task,
                            self.budget_accountant,
                            indices,
                            chunk_sizes,
                            self.probabilistic_cfg,
                        ),
                    )
                )
                processes[i].start()
            for i in range(num_processes):
                processes[i].join()
                return return_dict

        # Choose a planning method
        if self.config.planner.method == "min_cuts":
            return_dict = min_cuts()
        elif self.config.planner.method == "max_cuts":
            return_dict = max_cuts()
        elif self.config.planner.method == "optimal":
            return_dict = optimal_cuts()

        # Find the best solution
        plan = None
        best_objvalue = math.inf
        for _, (chunks, objvalue) in return_dict.items():
            if objvalue < best_objvalue:
                run_ops = []
                for chunk in chunks:
                    if isinstance(chunk, DeterministicChunk):
                        (i, j), noise_std = chunk
                        run_ops += [
                            RDet((i + task.blocks[0], j + task.blocks[0]), noise_std)
                        ]
                    elif isinstance(chunk, ProbabilisticChunk):
                        (i, j), (alpha, beta) = chunk
                        run_ops += [
                            RProb((i + task.blocks[0], j + task.blocks[0]), alpha, beta)
                        ]
                plan = A(l=run_ops, cost=objvalue)
                best_objvalue = objvalue
        return plan


def solve(
    kmin,
    kmax,
    cache_type,
    return_dict,
    cache,
    task,
    budget_accountant,
    indices,
    chunk_sizes,
    probabilistic_cfg,
):
    t = time.time()

    if cache_type == "DeterministicCache":
        for k in range(kmin, kmax + 1):
            ilp_args = ILPArgs(task, k, indices, cache, budget_accountant, chunk_sizes)
            chunks, objval = deterministic_optimize(ilp_args)
            if chunks:
                return_dict[k] = (chunks, objval)

    elif cache_type == "ProbabilisticCache":
        for k in range(kmin, kmax + 1):
            # Don't allow more than max_pmw_k PMW aggregations
            if k > probabilistic_cfg.max_pmw_k:
                break
            ilp_args = ILPArgs(task, k, indices, cache, budget_accountant, chunk_sizes)
            chunks, objval = probabilistic_optimize(ilp_args)
            if chunks:
                return_dict[k] = (chunks, objval)

    elif cache_type == "CombinedCache":
        best_objvalue = math.inf
        for k in range(kmin, kmax + 1):
            for k_prob in range(k + 1):
                # Don't allow more than max_pmw_k PMW aggregations
                if k_prob > probabilistic_cfg.max_pmw_k:
                    break
                k_det = k - k_prob
                for n_det in ["options here"]:
                    ilp_args = ILPArgs(
                        task, k, indices, cache, budget_accountant, chunk_sizes
                    )
                    chunks, objval = combined_optimize(n_det, k_det, k_prob, ilp_args)
                    if chunks and best_objvalue > objval:
                        # Return only one solution from this batch
                        return_dict[k] = (chunks, objval)
                        best_objvalue = objval

    t = (time.time() - t) / (kmax + 1 - kmin)
    # print(f"    Took:  {t}")


def deterministic_optimize(ilp_args):
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)
        m.Params.OutputFlag = 0

        # Initialize from args
        K = ilp_args.K  # Number of computations to be aggregated
        indices = ilp_args.indices
        blocks = list(range(ilp_args.task.blocks[0], ilp_args.task.blocks[1] + 1))
        a = ilp_args.task.utility
        b = ilp_args.task.utility_beta
        num_blocks = len(blocks)  # Number of computations to be aggregated
        n = ilp_args.chunk_sizes[
            (0, num_blocks - 1)
        ]  # Total size of data across all requested blocks

        run_budget_per_chunk = {}  # We want to minimize this
        noise_std_per_chunk = {}  # To return this for creating the plan

        # A variable per chunk
        x = m.addVars(
            [(i, j) for (i, j) in indices],
            vtype=GRB.INTEGER,
            lb=0,
            ub=1,
            name="x",
        )

        for (i, j) in indices:
            # Using the contents of the deterministic cache, check how much fresh budget
            # we need to spend across the blocks of the (i,j) chunk in order to reach noise-std.
            noise_std = deterministic_compute_utility_curve(
                a, b, n, ilp_args.chunk_sizes[(i, j)], K
            )
            noise_std_per_chunk[(i, j)] = noise_std

            run_budget = ilp_args.cache.deterministic_cache.estimate_run_budget(
                ilp_args.task.query_id,
                (blocks[i], blocks[j]),
                noise_std,
            )

            # Enough budget in blocks constraint to accommodate "run_budget"
            # If at least one block in chunk i,j does not have enough budget then Xij=0
            for k in range(i, j + 1):  # For every block in the chunk i,j
                if not ilp_args.budget_accountant.can_run(
                    (blocks[i], blocks[j]), run_budget
                ):
                    m.addConstr(x[i, j] == 0)
                    break

            # For the Objective Function let's reduce the renyi orders to one number by averaging them
            run_budget_per_chunk[(i, j)] = (
                sum(run_budget.epsilons) / len(run_budget.epsilons)
            ) * (j - i + 1)

        # No overlapping chunks constraint
        for k in range(num_blocks):  # For every block
            m.addConstr(
                (
                    gp.quicksum(
                        x[i, j]
                        for i in range(k + 1)
                        for j in range(k, num_blocks)
                        if (i, j) in indices
                    )
                )
                == 1
            )

        # Computations aggregated must be equal to K
        m.addConstr((gp.quicksum(x[i, j] for (i, j) in indices)) == K)

        # Objective function
        m.setObjective(x.prod(run_budget_per_chunk))
        m.ModelSense = GRB.MINIMIZE
        m.optimize()

        if m.status == GRB.OPTIMAL:
            chunks = []
            for (i, j) in indices:
                if int((abs(x[i, j].x - 1) < 1e-4)) == 1:
                    chunks.append(
                        DeterministicChunk((i, j), noise_std_per_chunk[(i, j)])
                    )
            return chunks, m.ObjVal
        return [], math.inf


def probabilistic_optimize(ilp_args):
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)
        m.Params.OutputFlag = 0

        # Initialize from args
        K = ilp_args.K  # Number of computations to be aggregated
        indices = ilp_args.indices
        blocks = list(range(ilp_args.task.blocks[0], ilp_args.task.blocks[1] + 1))
        a = ilp_args.task.utility
        b = ilp_args.task.utility_beta
        num_blocks = len(blocks)  # Number of computations to be aggregated

        run_budget_per_chunk = {}  # We want to minimize this
        alpha_beta_per_chunk = {}  # To return this for creating the plan

        # A variable per chunk
        x = m.addVars(
            [(i, j) for (i, j) in indices],
            vtype=GRB.INTEGER,
            lb=0,
            ub=1,
            name="x",
        )

        for (i, j) in indices:
            # Using the contents of the probabilistic cache, estimate how much fresh budget
            # we need to spend across the blocks of the (i,j) chunk in order to reach noise-std.
            alpha, beta = probabilistic_compute_utility_curve(a, b, K)
            alpha_beta_per_chunk[(i, j)] = (alpha, beta)

            run_budget = ilp_args.cache.probabilistic_cache.estimate_run_budget(
                ilp_args.task.query, (blocks[i], blocks[j]), alpha, beta
            )

            # Enough budget in blocks constraint to accommodate "run_budget"
            # If at least one block in chunk i,j does not have enough budget then Xij=0
            for k in range(i, j + 1):  # For every block in the chunk i,j
                if not ilp_args.budget_accountant.can_run(
                    (blocks[i], blocks[j]), run_budget
                ):
                    m.addConstr(x[i, j] == 0)
                    break

            # For the Objective Function let's reduce the renyi orders to one number by averaging them
            run_budget_per_chunk[(i, j)] = (
                sum(run_budget.epsilons) / len(run_budget.epsilons)
            ) * (j - i + 1)

        # No overlapping chunks constraint
        for k in range(num_blocks):  # For every block
            m.addConstr(
                (
                    gp.quicksum(
                        x[i, j]
                        for i in range(k + 1)
                        for j in range(k, num_blocks)
                        if (i, j) in indices
                    )
                )
                == 1
            )

        # Computations aggregated must be equal to K
        m.addConstr((gp.quicksum(x[i, j] for (i, j) in indices)) == K)

        # Objective function
        m.setObjective(x.prod(run_budget_per_chunk))
        m.ModelSense = GRB.MINIMIZE
        m.optimize()

        if m.status == GRB.OPTIMAL:
            chunks = []
            for (i, j) in indices:
                if int((abs(x[i, j].x - 1) < 1e-4)) == 1:
                    chunks.append(
                        ProbabilisticChunk((i, j), alpha_beta_per_chunk[(i, j)])
                    )
            return chunks, m.ObjVal
        return [], math.inf


def combined_optimize(n_det, k_det, k_prob, ilp_args):
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)
        m.Params.OutputFlag = 0

        # Initialize from args
        K = ilp_args.K  # Number of computations to be aggregated
        indices = ilp_args.indices
        blocks = list(range(ilp_args.task.blocks[0], ilp_args.task.blocks[1] + 1))
        a = ilp_args.task.utility
        b = ilp_args.task.utility_beta
        num_blocks = len(blocks)  # Number of computations to be aggregated
        n = ilp_args.chunk_sizes[
            (0, num_blocks - 1)
        ]  # Total size of data across all requested blocks

        b = 1 - math.sqrt(1 - b)  # b after union bound

        run_budget_per_det_chunk = {}
        run_budget_prob_per_chunk = {}
        noise_std_per_chunk = {}
        alpha_beta_per_chunk = {}

        # A variable per chunk for deterministic
        x = m.addVars(
            [(i, j) for (i, j) in indices],
            vtype=GRB.INTEGER,
            lb=0,
            ub=1,
            name="x",
        )

        # A variable per chunk for probabilistic
        y = m.addVars(
            [(i, j) for (i, j) in indices],
            vtype=GRB.INTEGER,
            lb=0,
            ub=1,
            name="y",
        )

        for (i, j) in indices:
            noise_std = deterministic_compute_utility_curve(
                a, b, n_det, ilp_args.chunk_sizes[(i, j)], k_det
            )
            alpha, beta = probabilistic_compute_utility_curve(a, b, k_prob)

            noise_std_per_chunk[(i, j)] = noise_std
            alpha_beta_per_chunk[(i, j)] = (alpha, beta)

            # Using the contents of the deterministic/probabilistic cache, check how much fresh budget
            # we need to spend across the blocks of the (i,j) chunk in order to reach noise-std
            run_budget_det = ilp_args.cache.deterministic_cache.estimate_run_budget(
                ilp_args.task.query_id,
                (blocks[i], blocks[j]),
                noise_std,
            )
            run_budget_prob = ilp_args.cache.probabilistic_cache.estimate_run_budget(
                ilp_args.task.query, (blocks[i], blocks[j]), alpha, beta
            )

            # Enough budget in blocks constraint to accommodate "run_budget"
            # If at least one block in chunk i,j does not have enough budget then Xij=0
            for k in range(i, j + 1):  # For every block in the chunk i,j
                if not ilp_args.budget_accountant.can_run(
                    (blocks[i], blocks[j]), run_budget_det
                ):
                    m.addConstr(x[i, j] == 0)
                    break
            for k in range(i, j + 1):  # For every block in the chunk i,j
                if not ilp_args.budget_accountant.can_run(
                    (blocks[i], blocks[j]), run_budget_prob
                ):
                    m.addConstr(y[i, j] == 0)
                    break

            # For the Objective Function let's reduce the renyi orders to one number by averaging them
            run_budget_per_det_chunk[(i, j)] = (
                sum(run_budget_det.epsilons) / len(run_budget_det.epsilons)
            ) * (j - i + 1)
            run_budget_prob_per_chunk[(i, j)] = (
                sum(run_budget_prob.epsilons) / len(run_budget_prob.epsilons)
            ) * (j - i + 1)

        # No overlapping chunks constraint
        for k in range(num_blocks):  # For every block
            m.addConstr(
                (
                    gp.quicksum(
                        x[i, j]
                        for i in range(k + 1)
                        for j in range(k, num_blocks)
                        if (i, j) in indices
                    )
                )
                == 1
            )
            m.addConstr(
                (
                    gp.quicksum(
                        y[i, j]
                        for i in range(k + 1)
                        for j in range(k, num_blocks)
                        if (i, j) in indices
                    )
                )
                == 1
            )
            for (i, j) in indices:
                m.addConstr(gp.quicksum(x[i, j] + y[i, j]) <= 1)

        # Computations aggregated must be equal to K_det/K_prob
        m.addConstr((gp.quicksum(x[i, j] for (i, j) in indices)) == k_det)
        m.addConstr((gp.quicksum(y[i, j] for (i, j) in indices)) == k_prob)

        # Objective function
        m.setObjective(
            x.prod(run_budget_per_det_chunk) + y.prod(run_budget_prob_per_chunk)
        )
        m.ModelSense = GRB.MINIMIZE
        m.optimize()

        if m.status == GRB.OPTIMAL:
            chunks = []
            for (i, j) in indices:
                if int((abs(x[i, j].x - 1) < 1e-4)) == 1:
                    chunks.append(
                        DeterministicChunk((i, j), noise_std_per_chunk[(i, j)])
                    )
                if int((abs(y[i, j].x - 1) < 1e-4)) == 1:
                    chunks.append(
                        ProbabilisticChunk((i, j), alpha_beta_per_chunk[(i, j)])
                    )
            return chunks, m.ObjVal
        return [], math.inf
