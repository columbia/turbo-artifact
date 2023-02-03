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
        if pmw.noise_std >= noise_std and pmw.pmw_updates_count <= pmw.heuristic_value:
            pmw.external_update(query=query, noisy_result=true_result + noise)

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
        elif not pmw:
            pmw = PMW(blocks, obj.alpha, obj.beta, **self.pmw_args)
            run_budget, worst_run_budget = pmw.estimate_run_budget(query)
        else:
            run_budget, worst_run_budget = pmw.estimate_run_budget(query)

        # TODO: This is leaking privacy, assume we have a good estimate already.
        return run_budget, worst_run_budget
