import math
import time
from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from termcolor import colored
from scipy.stats import rv_continuous
from precycle.budget import BasicBudget
from precycle.budget.curves import LaplaceCurve, PureDPtoRDP, ZeroCurve
from precycle.cache.exact_match_cache import CacheEntry
from precycle.utils.utils import get_blocks_size


class RunLaplace:
    def __init__(self, blocks, noise_std) -> None:
        self.blocks = blocks
        self.noise_std = noise_std
        self.epsilon = None

    def __str__(self):
        return f"RunLaplace({self.blocks}, {self.epsilon})"


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

    def run_laplace(self, run_op, query_id, query_db_format):
        node_size = get_blocks_size(run_op.blocks, self.config.blocks_metadata)
        sensitivity = 1 / node_size

        if self.config.exact_match_caching == False:
            # Run from scratch - don't look into the cache
            true_result = self.db.run_query(query_db_format, run_op.blocks)
            laplace_scale = run_op.noise_std / np.sqrt(2)
            epsilon = sensitivity / laplace_scale
            run_op.epsilon = epsilon
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
            laplace_scale = run_op.noise_std / np.sqrt(2)
            epsilon = sensitivity / laplace_scale
            run_op.epsilon = epsilon
            run_budget = (
                BasicBudget(epsilon)
                if self.config.puredp
                else LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
            )
            noise = np.random.laplace(scale=laplace_scale)

        else:  # Cached
            true_result = cache_entry.result

            if run_op.noise_std >= cache_entry.noise_std:
                # We already have a good estimate in the cache
                run_op.epsilon = 0
                run_budget = BasicBudget(0) if self.config.puredp else ZeroCurve()
                noise = cache_entry.noise
            else:
                # Do Variance Reduction using Kougofiannis et al: Privacy Relaxation
                # https://journalprivacyconfidentiality.org/index.php/jpc/article/view/649/632
                cached_laplace_scale = cache_entry.noise_std / np.sqrt(2)
                cached_pure_epsilon = sensitivity / cached_laplace_scale

                target_laplace_scale = run_op.noise_std / np.sqrt(2)
                target_pure_epsilon = sensitivity / target_laplace_scale

                # Run epsilon is the difference between the target_epsilon and cached_epsilon
                run_pure_epsilon = target_pure_epsilon - cached_pure_epsilon
                run_laplace_scale = sensitivity / run_pure_epsilon
                run_budget = (
                    BasicBudget(run_pure_epsilon)
                    if self.config.puredp
                    else LaplaceCurve(laplace_noise=run_laplace_scale / sensitivity)
                )

                # The new noise sample is computed as per Algorithm 1 in the above paper page 37
                noise = self.noise_down(cache_entry.noise, 1 / cached_laplace_scale, 1 / target_laplace_scale)
                print(colored(f"\n\n\ne_old: {cached_pure_epsilon}, e_new: {target_pure_epsilon}, old_noise: {cache_entry.noise}, new_noise: {noise} \n", "blue"))


        # If we used any fresh noise we need to update the cache
        if (not self.config.puredp and not isinstance(run_budget, ZeroCurve)) or (
            self.config.puredp and run_budget.epsilon > 0
        ):
            cache_entry = CacheEntry(
                result=true_result,
                noise_std=run_op.noise_std,  # It's the true std of our new linear combination
                noise=noise,
            )
            self.cache.exact_match_cache.write_entry(
                query_id, run_op.blocks, cache_entry
            )
        noisy_result = true_result + noise
        rv = RunReturnValue(true_result, noisy_result, run_budget)
        return rv

    # https://journalprivacyconfidentiality.org/index.php/jpc/article/view/649/632
    # https://gitlab.uwaterloo.ca/m2mazmud/cachedp-public/-/blob/main/apex/privacy/func/LCM_MP.py#L118
    def noise_down(self, old_noise, old_b_inv, new_b_inv):
        assert new_b_inv > old_b_inv
        sign = np.sign(old_noise)
        pdf = [old_b_inv / new_b_inv * np.exp((old_b_inv - new_b_inv) * abs(old_noise)),
                (new_b_inv - old_b_inv) / (2.0 * new_b_inv),
                (old_b_inv + new_b_inv) / (2.0 * new_b_inv) * (
                            1.0 - np.exp((old_b_inv - new_b_inv) * abs(old_noise)))]

        p = np.random.random()
        z = np.random.random()

        if p <= pdf[0]:
            new_noise = old_noise

        elif p <= pdf[0] + pdf[1]:
            new_noise = np.log(z) / (old_b_inv + new_b_inv)
            new_noise = new_noise * sign

        elif p <= pdf[0] + pdf[1] + pdf[2]:
            new_noise = np.log(
                z * (np.exp(abs(old_noise) * (old_b_inv - new_b_inv)) - 1.0) + 1.0) / (old_b_inv - new_b_inv)
            new_noise = new_noise * sign

        else:
            new_noise = abs(old_noise) - np.log(1.0 - z) / \
                        (new_b_inv + old_b_inv)
            new_noise = new_noise * sign
        return new_noise


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
