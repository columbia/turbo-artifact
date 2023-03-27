import math

from sortedcollections import OrderedSet

from precycle.executor import A, RunHistogram, RunLaplace, RunPMW
from precycle.planner.planner import Planner
from precycle.utils.utility_theorems import (
    get_epsilon_isotropic_laplace_concentration,
    get_epsilon_isotropic_laplace_monte_carlo,
    get_pmw_epsilon,
)
from precycle.utils.utils import get_blocks_size, satisfies_constraint


class MinCuts(Planner):
    def __init__(self, cache, budget_accountant, config):
        super().__init__(cache, budget_accountant, config)

    def get_min_cuts(self, blocks):
        """
        Returns the minimum number of nodes in the binary tree that can construct <blocks>
        """
        indices = OrderedSet()
        start, end = blocks
        chunk_end = start
        while chunk_end <= end:
            i = 1
            chunk_start = chunk_end
            while chunk_end <= end:
                x = chunk_start + 2**i - 1
                i += 1
                if x <= end and satisfies_constraint((chunk_start, x)):
                    chunk_end = x
                else:
                    indices.add((chunk_start, chunk_end))
                    chunk_end += 1
                    break
        return indices

    def get_execution_plan(self, task, force_laplace=False):
        """
        Picks a plan with minimal number of cuts that satisfies the binary constraint.
        If that plan can't be executed we don't look for another one
        """

        subqueries = self.get_min_cuts(task.blocks)
        n = get_blocks_size(task.blocks, self.config.blocks_metadata)
        k = len(subqueries)

        # NOTE: System wide accuracy for now
        alpha = self.config.alpha  # task.utility
        beta = self.config.beta  # task.utility_beta

        if self.mechanism_type == "Laplace" or force_laplace:
            min_epsilon = get_epsilon_isotropic_laplace_monte_carlo(
                alpha, beta, n, k, N=self.config.planner.monte_carlo_N
            )

            run_ops = []
            for (i, j) in subqueries:
                node_size = get_blocks_size((i, j), self.config.blocks_metadata)
                sensitivity = 1 / node_size
                laplace_scale = sensitivity / min_epsilon
                noise_std = math.sqrt(2) * laplace_scale
                run_ops += [RunLaplace((i, j), noise_std)]
            plan = A(l=run_ops, sv_check=False, cost=0)

        elif self.mechanism_type == "PMW":
            # TODO: Extend this to not work only in monoblock setting
            assert len(subqueries) == 1
            (i, j) = subqueries[0]
            node_size = get_blocks_size((i, j), self.config.blocks_metadata)
            epsilon = get_pmw_epsilon(alpha, beta, node_size)
            run_ops = [RunPMW((i, j), alpha, epsilon)]
            plan = A(l=run_ops, sv_check=False, cost=0)

        elif self.mechanism_type == "Hybrid":
            # Assign a Mechanism to each subquery
            # Using the Laplace Utility bound get the minimum epsilon that should be used by each subquery
            # In case a subquery is assigned to a Histogram run instead of a Laplace run
            # a final check must be done by a SV on the aggregated output to assess its quality.
            min_epsilon = get_epsilon_isotropic_laplace_monte_carlo(
                alpha, beta, n, k, N=self.config.planner.monte_carlo_N
            )

            sv_check = False
            run_ops = []
            for (i, j) in subqueries:
                # Measure the expected additional budget needed for a Laplace run.
                cache_entry = self.cache.exact_match_cache.read_entry(
                    task.query_id, (i, j)
                )
                node_size = get_blocks_size((i, j), self.config.blocks_metadata)
                sensitivity = 1 / node_size
                laplace_scale = sensitivity / min_epsilon
                noise_std = math.sqrt(2) * laplace_scale

                if (
                    (cache_entry and noise_std >= cache_entry.noise_std)
                ) or self.cache.histogram_cache.is_query_hard(task.query, (i, j)):
                    # If we have a good enough estimate in the cache choose Laplace because it will pay nothing.
                    # Also choose the Laplace if the histogram is not well trained according to our heuristic
                    run_ops += [RunLaplace((i, j), noise_std)]
                else:
                    sv_check = True
                    run_ops += [RunHistogram((i, j))]

            # TODO: before running the query check if there is enough budget
            # for it because we do not do the check here any more
            plan = A(l=run_ops, sv_check=sv_check, cost=0)

        # elif (
        #     self.mechanism_type == "Hybrid" and self.config.planner.monte_carlo == True
        # ):

        #     # Decide where to run the Laplace and where to run the Histogram
        #     # We look at the histogram heuristic only, not at the cache
        #     sv_check = False
        #     laplace_size = 0
        #     histogram_size = 0
        #     run_type: Dict[Tuple[int, int], str] = {}
        #     for (i, j) in subqueries:
        #         node_size = get_blocks_size((i, j), self.config.blocks_metadata)
        #         if self.cache.histogram_cache.is_query_hard(task.query, (i, j)):
        #             run_type[(i, j)] = "Laplace"
        #             laplace_size += node_size
        #         else:
        #             sv_check = True
        #             run_type[(i, j)] = "Histogram"
        #             #
        #             histogram_size += node_size

        #     # Split alpha and beta between both. Heuristic: weight by node size
        #     # It doesn't really matter when we do a global SV check
        #     # that covers the Laplace too (using too large alpha or beta will just result in more SV resets)
        #     # TODO: if Laplace are too noisy, you can increase alpha_laplace to have more precise PMW updates
        #     alpha_laplace = alpha * laplace_size / n
        #     beta_laplace = beta * laplace_size / n

        #     # print("alpha laplace", alpha_laplace)
        #     # print("beta laplace", beta_laplace)

        #     # NOTE: If you don't do global check, use triangle inequality for alpha and union bound for beta
        #     # and pass the remaining alpha/beta to the SV check

        #     # Instantiate the plan
        #     run_ops = []
        #     for (i, j) in subqueries:
        #         if run_type[(i, j)] == "Laplace":
        #             # The executor will look at the cache and compute the exact budget needed for each subquery
        #             run_ops += [
        #                 RunLaplace(
        #                     (i, j),
        #                     noise_std=None,
        #                     k=None,
        #                     alpha=alpha_laplace,
        #                     beta=beta_laplace,
        #                 )
        #             ]
        #         else:
        #             run_ops += [RunHistogram((i, j))]

        #     plan = A(l=run_ops, sv_check=sv_check, cost=0)

        return plan
