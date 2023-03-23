import math
import time
from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from termcolor import colored

from precycle.budget import BasicBudget
from precycle.budget.curves import LaplaceCurve, PureDPtoRDP, ZeroCurve
from precycle.cache.exact_match_cache import CacheEntry
from precycle.utils.utility_theorems import get_epsilon_vr_monte_carlo
from precycle.utils.utils import get_blocks_size


class RunLaplace:
    def __init__(
        self, blocks, noise_std, alpha=None, beta=None, n=None, k=None
    ) -> None:
        self.blocks = blocks
        self.noise_std = noise_std
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.k = k

    def __str__(self):
        return f"RunLaplace({self.blocks}, {self.noise_std})"


class RunLaplaceMonteCarlo:
    def __init__(self, blocks, epsilon):
        self.blocks = blocks
        self.epsilon = epsilon

    def __str__(self):
        return f"RunLaplaceMonteCarlo({self.blocks}, {self.epsilon})"


class RunHistogram:
    def __init__(self, blocks) -> None:
        self.blocks = blocks

    def __str__(self):
        return f"RunHistogram({self.blocks})"


class RunPMW:
    def __init__(self, blocks, alpha, epsilon) -> None:
        self.blocks = blocks
        self.alpha = alpha
        self.epsilon = epsilon

    def __str__(self):
        return f"RunPMW({self.blocks}, {self.alpha}, {self.epsilon})"


class A:
    def __init__(self, l, sv_check, cost=None) -> None:
        self.l = l
        self.cost = cost
        self.sv_check = sv_check

    def __str__(self):
        return f"Aggregate({[str(l) for l in self.l]})"


RunReturnValue = namedtuple(
    "RunReturnValue",
    [
        "true_result",
        "noisy_result",
        "run_budget",
    ],
)


class Executor:
    def __init__(self, cache, db, budget_accountant, config) -> None:
        self.db = db
        self.cache = cache
        self.config = config
        self.budget_accountant = budget_accountant

    def execute_plan(self, plan: A, task, run_metadata) -> Tuple[float, Dict]:
        total_size = 0
        true_result = None
        noisy_result = None
        status_message = None
        run_types = {}
        budget_per_block = {}
        true_partial_results = []
        noisy_partial_results = []

        if self.config.planner.monte_carlo:
            # Preprocess the plan: look at the cache for all Laplace, and *jointly* combine the optimal epsilon
            new_plan = []
            laplace_ops = []
            for run_op in plan.l:
                if isinstance(run_op, RunLaplace):
                    laplace_ops.append(run_op)
                else:
                    new_plan.append(run_op)

            montecarlo_laplace_ops = self.preprocess_montecarlo_laplace_ops(
                laplace_ops, query_id=task.query_id, N=self.config.planner.monte_carlo_N
            )
            new_plan.extend(montecarlo_laplace_ops)
            plan.l = new_plan

        logger.debug(f"Executing plan:\n{[str(op) for op in plan.l]}")

        for run_op in plan.l:
            if isinstance(run_op, RunLaplace):

                run_return_value = self.run_laplace(
                    run_op, task.query_id, task.query_db_format
                )
                run_types[str(run_op.blocks)] = "Laplace"

                # External Update to the Histogram
                # TODO: Add the convergence check, right now we have zero guarantees
                if self.config.mechanism.type == "Hybrid":
                    self.cache.histogram_cache.update_entry_histogram(
                        task.query,
                        run_op.blocks,
                        run_return_value.noisy_result,
                    )

            elif isinstance(run_op, RunLaplaceMonteCarlo):
                run_return_value = self.run_laplace_montecarlo(
                    run_op, task.query_id, task.query_db_format
                )

                # For the outside world it's just a Laplace
                run_types[str(run_op.blocks)] = "Laplace"

                # External update
                if self.config.mechanism.type == "Hybrid":
                    self.cache.histogram_cache.update_entry_histogram(
                        task.query,
                        run_op.blocks,
                        run_return_value.noisy_result,
                    )

            elif isinstance(run_op, RunHistogram):
                run_return_value = self.run_histogram(
                    run_op, task.query, task.query_db_format
                )
                run_types[str(run_op.blocks)] = "Histogram"

            elif isinstance(run_op, RunPMW):
                run_return_value = self.run_pmw(
                    run_op, task.query, task.query_db_format
                )
                run_types[str(run_op.blocks)] = "PMW"

            # Set run budgets for participating blocks
            for block in range(run_op.blocks[0], run_op.blocks[1] + 1):
                budget_per_block[block] = run_return_value.run_budget

            node_size = get_blocks_size(run_op.blocks, self.config.blocks_metadata)
            noisy_partial_results += [run_return_value.noisy_result * node_size]
            true_partial_results += [run_return_value.true_result * node_size]
            total_size += node_size

        if noisy_partial_results:
            # Aggregate outputs
            noisy_result = sum(noisy_partial_results) / total_size
            true_result = sum(true_partial_results) / total_size

            # TODO: do the check only on histogram partial results, not Direct Laplace ones
            # Do the final SV check if there is at least one Histogram run involved
            if plan.sv_check:
                status = self.run_sv_check(
                    noisy_result,
                    true_result,
                    task.blocks,
                    plan,
                    budget_per_block,
                    task.query,
                )
                if status == False:
                    # In case of failure we will try to run again the task
                    noisy_result = None
                    status_message = "sv_failed"
                    print("sv failed, task: ", task.id)
                run_metadata["sv_check_status"].append(status)
                sv_id = self.cache.sparse_vectors.get_lowest_common_ancestor(
                    task.blocks
                )
                run_metadata["sv_node_id"].append(sv_id)
            run_metadata["run_types"].append(run_types)
            run_metadata["budget_per_block"].append(budget_per_block)

            # Consume budget from blocks if necessary - we consume even if the check failed
            for block, run_budget in budget_per_block.items():
                # print(colored(f"Block: {block} - Budget: {run_budget.dump()}", "blue"))
                if (
                    not self.config.puredp and not isinstance(run_budget, ZeroCurve)
                ) or (self.config.puredp and run_budget.epsilon > 0):
                    self.budget_accountant.consume_block_budget(block, run_budget)

        return noisy_result, status_message

    def preprocess_montecarlo_laplace_ops(
        self, laplace_ops: List[RunLaplace], query_id: int, N: int = 100_000
    ) -> List[RunLaplaceMonteCarlo]:
        if len(laplace_ops) == 0:
            return []

        # Browse cache to populate state
        existing_epsilons = []
        chunk_sizes = []
        for run_op in laplace_ops:
            node_size = get_blocks_size(run_op.blocks, self.config.blocks_metadata)
            chunk_sizes.append(node_size)

            cache_entry = self.cache.exact_match_cache.read_entry(
                query_id, run_op.blocks
            )
            epsilons = (
                np.array(cache_entry.epsilons)
                if cache_entry is not None
                else np.array([])
            )
            existing_epsilons.append(epsilons)

        alphas = set(run_op.alpha for run_op in laplace_ops)
        betas = set(run_op.beta for run_op in laplace_ops)

        assert len(alphas) == 1, f"Alphas are not the same: {alphas}"
        assert len(betas) == 1, f"Betas are not the same: {betas}"

        alpha = alphas.pop()
        beta = betas.pop()

        # Drop some epsilons and use binary search monte carlo to find the best fresh epsilon
        fresh_epsilon, fresh_epsilon_mask = get_epsilon_vr_monte_carlo(
            existing_epsilons,
            chunk_sizes,
            alpha=alpha,
            beta=beta,
            N=N,
            n_processes=self.config.n_processes,
        )

        # Completely ignore the noise_std computed by the planner, it's just a loose upper bound
        laplace_montecarlo_ops = []
        for i, original_op in enumerate(laplace_ops):
            blocks = original_op.blocks
            epsilon = fresh_epsilon if fresh_epsilon_mask[i] else None
            laplace_montecarlo_ops.append(
                RunLaplaceMonteCarlo(blocks=blocks, epsilon=epsilon)
            )

        return laplace_montecarlo_ops

    def run_sv_check(
        self, noisy_result, true_result, blocks, plan, budget_per_block, query
    ):
        """
        1) Runs the SV check.
        2) Updates the run budgets for all blocks if SV uninitialized or for the blocks who haven't paid yet and arrived in the system if SV initialized.
        3) Flags the SV as uninitialized if check failed.
        4) Increases the heuristic threshold of participating histograms if check failed.
        """

        # Fetches the SV of the lowest common ancestor of <blocks>
        node_id = self.cache.sparse_vectors.get_lowest_common_ancestor(blocks)
        sv = self.cache.sparse_vectors.read_entry(node_id)
        if not sv:
            sv = self.cache.sparse_vectors.create_new_entry(node_id)

        # All blocks covered by the SV must pay
        blocks_to_pay = range(node_id[0], node_id[1] + 1)
        initialization_budget = (
            BasicBudget(3 * sv.epsilon)
            if self.config.puredp
            else PureDPtoRDP(epsilon=3 * sv.epsilon)
        )

        # Check if SV is initialized and set the initialization budgets to be consumed by blocks
        if not sv.initialized:
            sv.initialize()
            for block in blocks_to_pay:
                # If the block exists it has to pay, if it doesn't exist it will pay the first time the SV will be used again after it arrives
                if self.budget_accountant.get_block_budget(block) is not None:
                    if block not in budget_per_block:
                        budget_per_block[block] = initialization_budget
                    else:
                        budget_per_block[block] += initialization_budget
                        # print(budget_per_block)
                else:
                    # Set future block's outstanding payment
                    if block not in sv.outstanding_payment_blocks:
                        sv.outstanding_payment_blocks[block] = 0
                    sv.outstanding_payment_blocks[block] += 1
        else:
            # If it has been initialized but not all the blocks it covers had the opportunity to pay because they didn't exist yet let them pay now
            for block in blocks_to_pay:
                # If block hasn't paid yet and it now exists in the system make it pay now
                if (
                    block in sv.outstanding_payment_blocks
                    and self.budget_accountant.get_block_budget(block) is not None
                ):
                    # Make block pay for all outstanding payments
                    budget_per_block[block] = (
                        initialization_budget * sv.outstanding_payment_blocks[block]
                    )
                    del sv.outstanding_payment_blocks[block]

        # Now check whether we pass or fail the SV check
        if sv.check(true_result, noisy_result) == False:
            # Flag SV as uninitialized so that we pay again for its initialization next time we use it
            sv_check_status = False
            sv.initialized = False
            for run_op in plan.l:
                if isinstance(run_op, RunHistogram):
                    self.cache.histogram_cache.update_entry_threshold(
                        run_op.blocks, query
                    )
        else:
            sv_check_status = True
            print(colored("FREE LUNCH - yum yum\n", "blue"))

        self.cache.sparse_vectors.write_entry(sv)
        return sv_check_status

    def run_laplace_montecarlo(self, run_op, query_id, query_db_format):

        # Get the true result from the cache if possible
        cache_entry = self.cache.exact_match_cache.read_entry(query_id, run_op.blocks)
        if cache_entry is None:
            true_result = self.db.run_query(query_db_format, run_op.blocks)
            epsilons = []
            noises = []
        else:
            true_result = cache_entry.result
            epsilons = cache_entry.epsilons
            noises = cache_entry.noises

        # Just run a Laplace with the MonteCarlo epsilon, and store the result
        if run_op.epsilon is not None:
            node_size = get_blocks_size(run_op.blocks, self.config.blocks_metadata)
            sensitivity = 1 / node_size
            run_budget = (
                BasicBudget(run_op.epsilon)
                if self.config.puredp
                else LaplaceCurve(laplace_noise=run_op.epsilon)
            )
            fresh_noise = np.random.laplace(scale=run_op.epsilon / sensitivity)

            epsilons.append(run_op.epsilon)
            noises.append(fresh_noise)

            # Use variance reduction to compute the result
            sq_eps = np.array(epsilons) ** 2
            gammas = sq_eps / np.sum(sq_eps)
            noise_after_vr = np.dot(gammas, np.array(noises))

            # Variance of each individual Laplace (With the right senstivity)
            variances = sq_eps * 2 / node_size**2

            # Standard deviation of the linear combination
            std_after_vr = np.sqrt(np.dot(gammas**2, variances))

            # print(f"Std after MC VR: {std_after_vr}")

            # This is DP (even if we look at true_result internally) because sum_j gamma_j = 1
            noisy_result = true_result + noise_after_vr

            # Store the result in the cache
            cache_entry = CacheEntry(
                result=true_result,
                noise_std=std_after_vr,
                noise=noise_after_vr,
                epsilons=epsilons,
                noises=noises,
            )
            self.cache.exact_match_cache.write_entry(
                query_id, run_op.blocks, cache_entry
            )

        # MonteCarlo thinks the existing results are good enough
        else:
            print(f"MonteCarlo thinks the existing results are good enough")
            # TODO: If the cache is already good, do we even need to do VR again? No, if MC was used the whole time.
            run_budget = BasicBudget(0) if self.config.puredp else ZeroCurve()
            noisy_result = true_result + cache_entry.noise

        return RunReturnValue(true_result, noisy_result, run_budget)

    def run_laplace(self, run_op, query_id, query_db_format):
        node_size = get_blocks_size(run_op.blocks, self.config.blocks_metadata)
        sensitivity = 1 / node_size

        if self.config.exact_match_caching == False:
            # Run from scratch - don't look into the cache
            true_result = self.db.run_query(query_db_format, run_op.blocks)
            laplace_scale = run_op.noise_std / math.sqrt(2)
            epsilon = sensitivity / laplace_scale
            run_budget = (
                BasicBudget(epsilon)
                if self.config.puredp
                else LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
            )
            noise = np.random.laplace(scale=laplace_scale)
            noisy_result = true_result + noise
            rv = RunReturnValue(true_result, noisy_result, run_budget)
            return rv

        # Check for the entry inside the cache
        cache_entry = self.cache.exact_match_cache.read_entry(query_id, run_op.blocks)

        if not cache_entry:  # Not cached
            # True output never released except in debugging logs
            true_result = self.db.run_query(query_db_format, run_op.blocks)
            laplace_scale = run_op.noise_std / math.sqrt(2)
            epsilon = sensitivity / laplace_scale
            run_budget = (
                BasicBudget(epsilon)
                if self.config.puredp
                else LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
            )
            noise = np.random.laplace(scale=laplace_scale)
            epsilons = [epsilon]
            noises = [noise]

        else:  # Cached
            true_result = cache_entry.result
            epsilons = cache_entry.epsilons
            noises = cache_entry.noises

            if run_op.noise_std >= cache_entry.noise_std:
                # We already have a good estimate in the cache
                run_budget = BasicBudget(0) if self.config.puredp else ZeroCurve()
                noise = cache_entry.noise
            else:

                # Activate a complicated VR that is supposed to optimize variance but that fails sometimes
                SQRT_VR = False

                # We need to improve on the cache
                if not self.config.variance_reduction:
                    # Just compute from scratch and pay for it
                    laplace_scale = run_op.noise_std / math.sqrt(2)
                    epsilon = sensitivity / laplace_scale
                    run_budget = (
                        BasicBudget(epsilon)
                        if self.config.puredp
                        else LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
                    )
                    noise = np.random.laplace(scale=laplace_scale)

                    epsilons.append(epsilon)
                    noises.append(noise)

                elif SQRT_VR:
                    # noise_std = sqrt(2) / (node_size * epsilon)
                    target_laplace_scale = run_op.noise_std / math.sqrt(2)
                    k = run_op.k if run_op.k else 1  # Number of aggregations
                    target_pure_epsilon = sensitivity / target_laplace_scale

                    # print(
                    #     f"Starting VR with desired std {run_op.noise_std}, cache {cache_entry.noise_std}. epsilons={epsilons}, target pure epsilon={target_pure_epsilon}"
                    # )

                    # We want \sum \epsilon_i^2 = target_pure_epsilon^2
                    # The aggregation step will multiply by n_i/n

                    run_pure_epsilon = math.sqrt(
                        target_pure_epsilon**2 - sum([e**2 for e in epsilons])
                    )
                    run_laplace_scale = sensitivity / run_pure_epsilon
                    fresh_noise = np.random.laplace(scale=run_laplace_scale)
                    run_budget = (
                        BasicBudget(run_pure_epsilon)
                        if self.config.puredp
                        else LaplaceCurve(laplace_noise=run_laplace_scale / sensitivity)
                    )

                    epsilons.append(run_pure_epsilon)
                    noises.append(fresh_noise)

                    noise = sum(
                        n * (e**2 / target_pure_epsilon**2)
                        for n, e in zip(noises, epsilons)
                    )

                    # NOTE: the variance reduction lemma holds if max_ij b_ij is small enough over all the blocks.
                    # Conservative check: upper bound block by block, we lose a factor sqrt(k) but it's easier
                    # More general solution: do this check at the query level (not subqueries)
                    # We can optimize that if/when we implement Monte Carlo

                    beta = self.config.beta
                    if (
                        max(epsilons) * math.sqrt(math.log(2 / beta))
                        > math.sqrt(k) * target_pure_epsilon
                    ):
                        logger.warning("Conservative utility: compute from scratch")
                        # raise ValueError(
                        #     f"The utility theorem for multi-block VR breaks here (see Overleaf). \n\
                        #     espilons: {epsilons} target_pure_epsilon: {target_pure_epsilon} \n\
                        #     { max(epsilons) * math.sqrt(math.log(2 / beta))} > { math.sqrt(k) * target_pure_epsilon}\n\
                        #     Solution: drop the largest epsilon and repeat, or adjust the coefficients.
                        #       Or increase the target_pure_epsilon, or pay again for the same noise.
                        #       But let's see if this error ever happens in practice."
                        # Edit: it does happen often at the beginning.
                        # )

                        # TODO: just use Monte Carlo instead of this weird conditional bound
                        # Conservative solution: ignore the cache and compute from scratch

                        epsilons.pop(-1)
                        noises.pop(-1)

                        laplace_scale = run_op.noise_std / math.sqrt(2)
                        epsilon = sensitivity / laplace_scale
                        run_budget = (
                            BasicBudget(epsilon)
                            if self.config.puredp
                            else LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
                        )
                        noise = np.random.laplace(scale=laplace_scale)

                        epsilons.append(epsilon)
                        noises.append(noise)

                else:
                    # Recover epsilon with noise_std = sqrt(2) / (node_size * epsilon)
                    target_laplace_scale = run_op.noise_std / math.sqrt(2)
                    k = run_op.k if run_op.k else 1  # Number of aggregations
                    target_pure_epsilon = sensitivity / target_laplace_scale

                    # We already have j-1 epsilons, our goal is to have j identical (scaled) Laplace
                    j = len(epsilons) + 1

                    # Linear combination coefficients
                    gammas = [
                        e / (math.sqrt(j) * target_pure_epsilon) for e in epsilons
                    ]
                    x = 1 - sum(gammas)

                    # Check that the multiblock VR concentration bound holds
                    if x < 0 or k * j < math.log(2 / self.config.beta):
                        # logger.warning(
                        # f"Fallback on the b_M branch for Vr: x = {x} < 0, or {k} * {j} < {math.log(2 / self.config.beta)} "
                        # )

                        # The concentration bound doesn't hold, probably because we don't have enough Laplace
                        # In particular k < math.log(2 / self.config.beta) so `get_laplace_epsilon` is on the b_M branch
                        # We could just take one Laplace with the right b_M

                        laplace_scale = run_op.noise_std / math.sqrt(2)
                        epsilon = sensitivity / laplace_scale
                        run_budget = (
                            BasicBudget(epsilon)
                            if self.config.puredp
                            else LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
                        )
                        noise = np.random.laplace(scale=laplace_scale)

                        epsilons.append(epsilon)
                        noises.append(noise)
                    else:
                        # logger.info("Doing VR just fine.")

                        # Compute the new noise
                        run_pure_epsilon = x * math.sqrt(j) * target_pure_epsilon
                        run_laplace_scale = sensitivity / run_pure_epsilon
                        fresh_noise = np.random.laplace(scale=run_laplace_scale)
                        run_budget = (
                            BasicBudget(run_pure_epsilon)
                            if self.config.puredp
                            else LaplaceCurve(
                                laplace_noise=run_laplace_scale / sensitivity
                            )
                        )

                        epsilons.append(run_pure_epsilon)
                        noises.append(fresh_noise)
                        gammas.append(x)  # sum(gammas) = 1 now

                        noise = sum(n * g for n, g in zip(noises, gammas))

        # If we used any fresh noise we need to update the cache
        if (not self.config.puredp and not isinstance(run_budget, ZeroCurve)) or (
            self.config.puredp and run_budget.epsilon > 0
        ):

            cache_entry = CacheEntry(
                result=true_result,
                noise_std=run_op.noise_std,  # It's the true std of our new linear combination
                noise=noise,
                epsilons=epsilons,
                noises=noises,
            )
            self.cache.exact_match_cache.write_entry(
                query_id, run_op.blocks, cache_entry
            )
        noisy_result = true_result + noise
        rv = RunReturnValue(true_result, noisy_result, run_budget)
        return rv

    def run_histogram(self, run_op, query, query_db_format):
        cache_entry = self.cache.histogram_cache.read_entry(run_op.blocks)
        if not cache_entry:
            cache_entry = self.cache.histogram_cache.create_new_entry(run_op.blocks)
            self.cache.histogram_cache.write_entry(run_op.blocks, cache_entry)

        # True output never released except in debugging logs
        true_result = self.db.run_query(query_db_format, run_op.blocks)
        # Run histogram to get the predicted output
        noisy_result = cache_entry.histogram.run(query)
        # Histogram prediction doesn't cost anything
        run_budget = BasicBudget(0) if self.config.puredp else ZeroCurve()

        rv = RunReturnValue(true_result, noisy_result, run_budget)
        return rv

    def run_pmw(self, run_op, query, query_db_format):
        pmw = self.cache.pmw_cache.get_entry(run_op.blocks)
        if not pmw:
            pmw = self.cache.pmw_cache.add_entry(run_op.blocks)

        # True output never released except in debugging logs
        true_result = self.db.run_query(query_db_format, run_op.blocks)

        # We can't run a powerful query using a weaker PMW
        assert run_op.alpha <= pmw.alpha
        assert run_op.epsilon <= pmw.epsilon

        noisy_result, run_budget, _ = pmw.run(query, true_result)
        rv = RunReturnValue(true_result, noisy_result, run_budget)
        return rv
