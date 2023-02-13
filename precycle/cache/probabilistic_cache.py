from loguru import logger
from precycle.cache.pmw import PMW
from precycle.cache.cache import Cache
from copy import copy


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

    def add_entry(self, blocks, alpha, beta, old_pmw=None):
        pmw = PMW(blocks, alpha, beta, old_pmw, **self.pmw_args)
        self.key_values[blocks] = pmw
        return pmw

    def get_entry(self, blocks):
        if blocks in self.key_values:
            return self.key_values[blocks]
        return None

    def update_entry(self, query, blocks, true_result, noise_std, noise):
        pmw = self.get_entry(blocks)
        if (
            not pmw
        ):  # If there is no PMW for the blocks then create it with the default config
            pmw = self.add_entry(
                blocks, self.pmw_accuracy.alpha, self.pmw_accuracy.beta
            )
        # If the new entry is good enough for the PMW
        # TODO: Enabled External Updates here.
        # update if the entry's noise std is at most bigger by 0.001
        if (
            pmw.noise_std >= noise_std - 0.001
        ):  # and pmw.pmw_updates_count <= pmw.heuristic_value:
            pmw.external_update(query=query, noisy_result=true_result + noise)
        else:
            print(f"External update Failed. Not accurate enough result. {noise_std} > {pmw.noise_std}")

            logger.error(
                f"External update Failed. Not accurate enough result. {noise_std} > {pmw.noise_std}"
            )
            exit(1)

    def estimate_run_budget(self, query, blocks, alpha, beta):
        """
        Checks the cache and returns the budget we need to spend if we want to run this query with given accuracy guarantees.
        """
        pmw = self.get_entry(blocks)
        obj = pmw if pmw else self.pmw_accuracy

        if alpha > obj.alpha or beta < obj.beta:
            pmw = PMW(blocks, alpha, beta, **self.pmw_args)
            run_budget, worst_run_budget = pmw.estimate_run_budget(query)
            logger.error(
                "Plan requires more powerful PMW than the one cached. We decided this wouldn't happen."
            )
            exit(1)
        elif not pmw:
            pmw = PMW(blocks, obj.alpha, obj.beta, **self.pmw_args)
            run_budget, worst_run_budget = pmw.estimate_run_budget(query)
        else:
            run_budget, worst_run_budget = pmw.estimate_run_budget(query)

        # TODO: This is leaking privacy, assume we have a good estimate already.
        return run_budget, worst_run_budget

    def is_query_hard_on_pmw(self, query, blocks):
        """
        Checks the cache and returns whether the query will be hard for a PMW on <blocks>
        """
        pmw = self.get_entry(blocks)
        if not pmw:
            pmw = PMW(
                blocks, self.pmw_accuracy.alpha, self.pmw_accuracy.beta, **self.pmw_args
            )
        return pmw.is_query_hard(query)
