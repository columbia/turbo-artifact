from copy import copy

from loguru import logger

from precycle.cache.cache import Cache
from precycle.cache.pmw import PMW
from precycle.utils.compute_utility_curve import probabilistic_compute_utility_curve
from precycle.utils.utils import get_blocks_size


class ProbabilisticCache(Cache):
    def __init__(self, config):
        raise NotImplementedError


class MockProbabilisticCache(Cache):
    def __init__(self, config):
        self.key_values = {}
        self.config = config
        self.pmw_args = copy(config.cache.pmw_cfg)
        self.pmw_args.update({"blocks_metadata": self.config.blocks_metadata})
        self.pmw_accuracy = self.config.cache.pmw_accuracy
        self.block_metadata = self.config.blocks_metadata

    # def warmup(self, blocks):
    # Given a new pmw on <blocks>

    def add_entry(self, blocks):
        alpha = self.pmw_accuracy.alpha
        beta = self.pmw_accuracy.beta
        max_pmw_k = self.pmw_accuracy.max_pmw_k

        # n = (blocks[1] - blocks[0] + 1) * self.block_size
        # alpha, nu = probabilistic_compute_utility_curve(alpha, beta, n, max_pmw_k)
        # pmw = PMW(blocks, alpha, nu, **self.pmw_args)

        n = get_blocks_size(blocks, self.blocks_metadata)
        epsilon = 1
        # TODO: implement utility

        pmw = PMW(
            alpha=alpha,
            epsilon=epsilon,
            n=n,
            id=str(blocks)[1:-1].replace(", ", "-"),
            domain_size=self.blocks_metadata["domain_size"],
        )

        self.key_values[blocks] = pmw
        return pmw

    def get_entry(self, blocks):
        if blocks in self.key_values:
            return self.key_values[blocks]
        return None

    def update_entry(self, query, blocks, true_result, noise_std, alpha, beta, noise):
        pmw = self.get_entry(blocks)
        if not pmw:
            pmw = self.add_entry(blocks)
        # Relaxed condition for External Update - if I was to compare the noise_std I would rarely update in the multiblock case
        if pmw.alpha >= alpha:
            pmw.external_update(query=query, noisy_result=true_result + noise)
        else:
            logger.error(
                f"External update Failed. Not accurate enough result. {alpha} > {pmw.alpha}"
            )
            exit(1)

        # if (
        #     pmw.noise_std >= noise_std
        # ):  # and pmw.pmw_updates_count <= pmw.heuristic_value:
        #     pmw.external_update(query=query, noisy_result=true_result + noise)
        # else:
        #     logger.error(
        #         f"External update Failed. Not accurate enough result. {noise_std} > {pmw.noise_std}"
        #     )
        #     exit(1)

    def is_query_hard_on_pmw(self, query, blocks):
        """
        Checks the cache and returns whether the query will be hard for a PMW on <blocks>
        """
        # if blocks[1]-blocks[0]+1 < 4:
        # return True
        pmw = self.get_entry(blocks)
        return True if not pmw else pmw.is_query_hard(query)

    # def estimate_run_budget(self, query, blocks, alpha, beta):
    #     """
    #     Checks the cache and returns the budget we need to spend if we want to run this query with given accuracy guarantees.
    #     """
    #     pmw = self.get_entry(blocks)
    #     obj = pmw if pmw else self.pmw_accuracy

    #     if alpha > obj.alpha or beta < obj.beta:
    #         pmw = PMW(blocks, alpha, beta, **self.pmw_args)
    #         run_budget, worst_run_budget = pmw.estimate_run_budget(query)
    #         logger.error(
    #             "Plan requires more powerful PMW than the one cached. We decided this wouldn't happen."
    #         )
    #         exit(1)
    #     elif not pmw:
    #         pmw = PMW(blocks, obj.alpha, obj.beta, **self.pmw_args)
    #         run_budget, worst_run_budget = pmw.estimate_run_budget(query)
    #     else:
    #         run_budget, worst_run_budget = pmw.estimate_run_budget(query)

    #     # TODO: This is leaking privacy, assume we have a good estimate already.
    #     return run_budget, worst_run_budget
