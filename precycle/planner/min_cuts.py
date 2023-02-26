import math
from sortedcollections import OrderedSet
from precycle.planner.planner import Planner
from precycle.utils.utils import satisfies_constraint
from precycle.executor import A, RunLaplace, RunHistogram, RunPMW
from precycle.utils.utility_theorems import get_laplace_epsilon, get_pmw_epsilon


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

    def get_execution_plan(self, task):
        """
        Picks a plan with minimal number of cuts that satisfies the binary constraint.
        If that plan can't be executed we don't look for another one
        """

        subqueries = self.get_min_cuts(task.blocks)
        block_size = self.config.blocks_metadata["block_size"]
        num_blocks = task.blocks[1] - task.blocks[0] + 1
        n = num_blocks * block_size

        # NOTE: System wide accuracy for now
        alpha = self.config.alpha  # task.utility
        beta = self.config.beta  # task.utility_beta

        if self.cache_type == "LaplaceCache":
            run_ops = []
            min_epsilon = get_laplace_epsilon(alpha, beta, n, len(subqueries))
            for (i, j) in subqueries:
                node_size = (j - i + 1) * block_size
                sensitivity = 1 / node_size
                laplace_scale = sensitivity / min_epsilon
                noise_std = math.sqrt(2) * laplace_scale
                run_ops += [RunLaplace((i, j), noise_std)]
            plan = A(l=run_ops, sv_check=False, cost=0)

        elif self.cache_type == "PMWCache":
            # NOTE: This is PMW.To be used only in the Monoblock case
            assert len(subqueries) == 1
            (i, j) = subqueries[0]
            epsilon = get_pmw_epsilon(alpha, beta, (j - i + 1) * block_size, 1)
            print(block_size, epsilon)
            run_ops = [RunPMW((i, j), alpha, epsilon)]
            plan = A(l=run_ops, sv_check=False, cost=0)

        elif self.cache_type == "HybridCache":
            # Assign a Mechanism to each subquery
            # Using the Laplace Utility bound get the minimum epsilon that should be used by each subquery
            # In case a subquery is assigned to a Histogram run instead of a Laplace run, then this epsilon
            # might not have been enough and a final check must be done by a SV on the aggregated output to assess its quality.
            min_epsilon = get_laplace_epsilon(alpha, beta, n, len(subqueries))
            sv_check = False
            run_ops = []
            for (i, j) in subqueries:
                # Measure the expected additional budget needed for a Laplace run.
                cache_entry = self.cache.laplace_cache.read_entry(task.query_id, (i, j))
                node_size = (j - i + 1) * block_size
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

        return plan
