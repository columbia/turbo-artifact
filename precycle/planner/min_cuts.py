import os
import math
from sortedcollections import OrderedSet
from precycle.executor import A, RDet, RProb
from precycle.planner.planner import Planner
import time

# from precycle.utils.utils import get_blocks_size
from precycle.utils.compute_utility_curve import (
    deterministic_compute_utility_curve,
    probabilistic_compute_utility_curve,
)


class MinCuts(Planner):
    def __init__(self, cache, budget_accountant, config):
        super().__init__(cache, budget_accountant, config)

    def satisfies_constraint(self, blocks, branching_factor=2):
        """
        Checks if <blocks> satisfies the binary structure constraint
        """
        n = blocks[1] - blocks[0] + 1
        if not math.log(n, branching_factor).is_integer():
            return False
        if (blocks[0] % n) != 0:
            return False
        return True

    def get_min_cuts(self, blocks):
        """
        Returns the minimum number of nodes in the binary tree that can construct <blocks>
        """
        indices = OrderedSet()
        start, end = blocks
        n = end - start + 1
        chunk_end = start
        while chunk_end <= end:
            i = 1
            chunk_start = chunk_end
            while chunk_end <= end:
                x = chunk_start + 2**i - 1
                i += 1
                if x <= end and self.satisfies_constraint((chunk_start, x)):
                    chunk_end = x
                else:
                    indices.add((chunk_start, chunk_end))
                    chunk_end += 1
                    break
        return indices

    def get_execution_plan(self, task):
        """
        Picks a plan with minimal number of cuts that satisfies the binary constraint.
        If that plan can't be executed we don't look for another one
        """

        subqueries = self.get_min_cuts(task.blocks)
        a = task.utility
        b = task.utility_beta

        if self.cache_type == "DeterministicCache":
            block_size = self.config.blocks_metadata["block_size"]
            num_blocks = task.blocks[1] - task.blocks[0] + 1
            n = num_blocks * block_size
            run_ops = []
            for (i, j) in subqueries:
                noise_std = deterministic_compute_utility_curve(
                    a, b, n, (j - i + 1) * block_size, len(subqueries)
                )
                run_ops += [RDet((i, j), noise_std)]
            plan = A(l=run_ops, cost=0)

        elif self.cache_type == "ProbabilisticCache":
            # PMW computations must not exceed threshold otherwise we will break accuracy
            if len(subqueries) > self.config.cache.probabilistic_cfg.max_pmw_k:
                return None

            block_size = self.config.blocks_metadata["block_size"]
            num_blocks = task.blocks[1] - task.blocks[0] + 1
            n = num_blocks * block_size

            run_ops = []
            for (i, j) in subqueries:
                # Compute alpha, nu for each pmw run
                alpha, nu = probabilistic_compute_utility_curve(
                    a, b, (j - i + 1) * block_size, len(subqueries)
                )
                run_ops += [RProb((i, j), alpha, nu)]
            plan = A(l=run_ops, cost=0)

        else:
            # Assign a Mechanism to each subquery
            pmw_nodes = []
            laplace_nodes = []
            for subquery in subqueries:
                # Don't use PMW if query is hard for it, run using a simple Laplace mechanism instead
                obj = (
                    laplace_nodes
                    if self.cache.probabilistic_cache.is_query_hard_on_pmw(
                        task.query, task.blocks
                    )
                    else pmw_nodes
                )
                obj.append(subquery)

            pmw_nodes_len = len(pmw_nodes)
            laplace_nodes_len = len(laplace_nodes)

            # PMW computations must not exceed threshold otherwise we will break accuracy
            if pmw_nodes_len > self.config.cache.probabilistic_cfg.max_pmw_k:
                return None

            # Now that each subquery is assigned to a mechanism determine
            # the run budgets using the utility theorems
            if pmw_nodes_len > 0 and laplace_nodes_len > 0:
                # Union bound -> decrease b
                b = 1 - math.sqrt(1 - b)

            block_size = self.config.blocks_metadata["block_size"]

            n_laplace = 0
            for (i, j) in laplace_nodes:
                n_laplace += (j - i + 1) * block_size

            # Create the plan
            run_ops = []
            for (i, j) in laplace_nodes:
                # Compute noise scale for the laplace runs
                noise_std = deterministic_compute_utility_curve(
                    a, b, n_laplace, (j - i + 1) * block_size, laplace_nodes_len
                )
                run_ops += [RDet((i, j), noise_std)]

            for (i, j) in pmw_nodes:
                # Compute alpha, nu for each pmw run
                alpha, nu = probabilistic_compute_utility_curve(
                    a, b, (j - i + 1) * block_size, pmw_nodes_len
                )
                run_ops += [RProb((i, j), alpha, nu)]

            # TODO: before running the query check if there is enough budget
            # for it because we do not do the check here any more
            plan = A(l=run_ops, cost=0)

            # if pmw_nodes_len > 0 and laplace_nodes_len > 0:
            # time.sleep(4)

        return plan
