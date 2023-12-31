import time
from collections import namedtuple
from typing import Dict, Tuple

import numpy as np
from loguru import logger
from termcolor import colored

from turbo.budget import BasicBudget
from turbo.budget.curves import LaplaceCurve, PureDPtoRDP, ZeroCurve
from turbo.cache.exact_match_cache import CacheEntry
from turbo.utils.utils import (
    HISTOGRAM_RUNTYPE,
    LAPLACE_RUNTYPE,
    PMW_RUNTYPE,
    get_blocks_size,
    get_node_key,
    mlflow_log,
)


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
        self.count = 0

    def execute_plan(self, plan: A, task, run_metadata) -> Tuple[float, Dict]:
        total_size = 0
        true_result = None
        noisy_result = None
        status_message = None
        run_types = {}
        budget_per_block = {}
        true_partial_results = []
        noisy_partial_results = []

        logger.debug(f"Executing plan:\n{[str(op) for op in plan.l]}")

        node_sizes = {}
        laplace_hits = {}
        pmw_hits = {}
        external_updates = {}
        db_runtime = {}

        for run_op in plan.l:

            node_key = get_node_key(run_op.blocks)

            if isinstance(run_op, RunLaplace):
                cached_true_result = None
                if node_key in run_metadata["true_result_per_node"]:
                    cached_true_result = run_metadata["true_result_per_node"][node_key]

                run_return_value, run_laplace_metadata = self.run_laplace(
                    run_op, task.query_id, task.query_db_format, cached_true_result
                )
                run_types[node_key] = LAPLACE_RUNTYPE
                laplace_hits[node_key] = run_laplace_metadata.get("hit", 0)
                db_runtime[node_key] = run_laplace_metadata.get("db_runtime", 0)

                # External Update to the Histogram (will do the check inside)
                if self.config.mechanism.type == "Hybrid":
                    update = self.cache.histogram_cache.update_entry_histogram(
                        task.query,
                        run_op.blocks,
                        run_return_value.noisy_result,
                        epsilon=run_op.epsilon,
                    )
                    external_updates[node_key] = update

            elif isinstance(run_op, RunHistogram):
                run_return_value, run_histogram_metadata = self.run_histogram(
                    run_op, task.query, task.query_db_format
                )
                run_types[node_key] = HISTOGRAM_RUNTYPE
                db_runtime[node_key] = run_histogram_metadata.get("db_runtime", 0)

            elif isinstance(run_op, RunPMW):
                run_return_value, pmw_metadata = self.run_pmw(
                    run_op, task.query, task.query_db_format
                )
                run_types[node_key] = PMW_RUNTYPE
                pmw_hits[node_key] = 0 if pmw_metadata["hard_query"] else 1

            # Set run budgets for participating blocks
            for block in range(run_op.blocks[0], run_op.blocks[1] + 1):
                budget_per_block[block] = run_return_value.run_budget

            node_size = get_blocks_size(run_op.blocks, self.config.blocks_metadata)
            node_sizes[node_key] = node_size

            noisy_partial_results += [run_return_value.noisy_result * node_size]
            true_partial_results += [run_return_value.true_result * node_size]
            total_size += node_size

            run_metadata["true_result_per_node"][
                node_key
            ] = run_return_value.true_result

        # Modify metadata inplace (a bit ugly)
        run_metadata["node_sizes"] = node_sizes
        run_metadata["total_size"] = total_size

        # We append to the metadata because after SV resets we need to repeat the same query
        run_metadata["laplace_hits"].append(laplace_hits)
        run_metadata["pmw_hits"].append(pmw_hits)
        run_metadata["external_updates"].append(external_updates)
        run_metadata["db_runtimes"].append(db_runtime)

        if noisy_partial_results:
            # Aggregate outputs
            noisy_result = sum(noisy_partial_results) / total_size
            true_result = sum(true_partial_results) / total_size
            logger.debug(
                "noisy",
                noisy_result,
                "true",
                true_result,
                "err",
                true_result - noisy_result,
            )
            run_metadata["error"] = true_result - noisy_result

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
                    logger.debug("sv failed, task: ", task.id)
                run_metadata["sv_check_status"].append(status)
                sv_id = task.blocks
                run_metadata["sv_node_id"].append(sv_id)
            run_metadata["run_types"].append(run_types)
            run_metadata["budget_per_block"].append(budget_per_block)
            # run_metadata["heuristic_update_runtime"].append(heuristic_runtime)

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

        sv = self.cache.sparse_vectors.read_entry(blocks)
        if not sv:
            sv = self.cache.sparse_vectors.create_new_entry(blocks)

        # All blocks covered by the SV must pay
        blocks_to_pay = range(blocks[0], blocks[1] + 1)
        initialization_budget = (
            BasicBudget(3 * sv.epsilon)
            if self.config.puredp
            else PureDPtoRDP(epsilon=3 * sv.epsilon)
        )

        # Check if SV is initialized and set the initialization budgets to be consumed by blocks
        if not sv.initialized:
            sv.initialize()
            for block in blocks_to_pay:
                if block not in budget_per_block:
                    budget_per_block[block] = initialization_budget
                else:
                    budget_per_block[block] += initialization_budget
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
            logger.debug("FREE LUNCH - yum yum\n", "blue")

        self.cache.sparse_vectors.write_entry(sv)
        return sv_check_status

    def run_laplace(self, run_op, query_id, query_db_format, cached_true_result):

        run_laplace_metadata = {}

        node_size = get_blocks_size(run_op.blocks, self.config.blocks_metadata)
        sensitivity = 1 / node_size

        if self.config.exact_match_caching == False:
            # Run from scratch - don't look into the cache
            start_time = time.time()
            true_result = (
                self.db.run_query(query_db_format, run_op.blocks)
                if cached_true_result is None
                else cached_true_result
            )
            run_laplace_metadata["hit"] = 0
            run_laplace_metadata["db_runtime"] = time.time() - start_time

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
            return rv, run_laplace_metadata

        # Check for the entry inside the cache
        cache_entry = self.cache.exact_match_cache.read_entry(query_id, run_op.blocks)

        if not cache_entry:  # Not cached
            # True output never released except in debugging logs
            start_time = time.time()
            true_result = (
                self.db.run_query(query_db_format, run_op.blocks)
                if cached_true_result is None
                else cached_true_result
            )
            run_laplace_metadata["db_runtime"] = time.time() - start_time

            laplace_scale = run_op.noise_std / np.sqrt(2)
            epsilon = sensitivity / laplace_scale
            run_op.epsilon = epsilon
            run_budget = (
                BasicBudget(epsilon)
                if self.config.puredp
                else LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
            )
            noise = np.random.laplace(scale=laplace_scale)

            run_laplace_metadata["hit"] = 0

        else:  # Cached
            true_result = cache_entry.result

            if run_op.noise_std >= cache_entry.noise_std:
                # We already have a good estimate in the cache
                laplace_scale = run_op.noise_std / np.sqrt(2)
                epsilon = sensitivity / laplace_scale

                if (
                    self.config.mechanism.probabilistic_cfg.external_update_on_cached_results
                ):
                    run_op.epsilon = epsilon
                else:
                    run_op.epsilon = 0
                run_budget = BasicBudget(0) if self.config.puredp else ZeroCurve()
                noise = cache_entry.noise

                run_laplace_metadata["hit"] = 1

            else:
                laplace_scale = run_op.noise_std / np.sqrt(2)
                epsilon = sensitivity / laplace_scale
                run_op.epsilon = epsilon
                run_budget = (
                    BasicBudget(epsilon)
                    if self.config.puredp
                    else LaplaceCurve(laplace_noise=laplace_scale / sensitivity)
                )
                noise = np.random.laplace(scale=laplace_scale)

                # NOTE: if you do VR you can use hit = (new_eps - old_eps) / old_eps
                run_laplace_metadata["hit"] = 0

        # If we used any fresh noise we need to update the cache
        if (not self.config.puredp and not isinstance(run_budget, ZeroCurve)) or (
            self.config.puredp and run_budget.epsilon > 0
        ):
            cache_entry = CacheEntry(
                result=true_result,
                noise_std=run_op.noise_std,
                noise=noise,
            )
            self.cache.exact_match_cache.write_entry(
                query_id, run_op.blocks, cache_entry
            )
        noisy_result = true_result + noise
        rv = RunReturnValue(true_result, noisy_result, run_budget)
        return rv, run_laplace_metadata

    def run_histogram(self, run_op, query, query_db_format):
        run_histogram_metadata = {}

        cache_entry = self.cache.histogram_cache.read_entry(run_op.blocks)
        if not cache_entry:
            cache_entry = self.cache.histogram_cache.create_new_entry(run_op.blocks)
            self.cache.histogram_cache.write_entry(run_op.blocks, cache_entry)

        # True output never released except in debugging logs
        start_time = time.time()
        true_result = self.db.run_query(query_db_format, run_op.blocks)
        run_histogram_metadata["db_runtime"] = time.time() - start_time

        # Run histogram to get the predicted output
        noisy_result = cache_entry.histogram.run(query)
        # Histogram prediction doesn't cost anything
        run_budget = BasicBudget(0) if self.config.puredp else ZeroCurve()

        rv = RunReturnValue(true_result, noisy_result, run_budget)
        return rv, run_histogram_metadata

    def run_pmw(self, run_op, query, query_db_format):
        pmw = self.cache.pmw_cache.get_entry(run_op.blocks)
        if not pmw:
            pmw = self.cache.pmw_cache.add_entry(run_op.blocks)

        # True output never released except in debugging logs
        true_result = self.db.run_query(query_db_format, run_op.blocks)

        # We can't run a powerful query using a weaker PMW
        assert run_op.alpha <= pmw.alpha
        assert run_op.epsilon <= pmw.epsilon

        noisy_result, run_budget, run_metadata = pmw.run(query, true_result)
        rv = RunReturnValue(true_result, noisy_result, run_budget)
        return rv, run_metadata
