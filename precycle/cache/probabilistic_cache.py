import math
from precycle.cache.pmw import PMW
from precycle.cache.cache import Cache
from precycle.utils.utils import get_blocks_size


class ProbabilisticCache(Cache):
    def __init__(self, config):
        raise NotImplementedError


class MockProbabilisticCache(Cache):
    def __init__(self, config):
        self.key_values = {}
        self.config = config
        self.pmw_args = config.cache.pmw_cfg

    def add_entry(self, blocks):
        pmw = PMW(blocks, self.config.blocks_metadata, **self.pmw_args)
        self.key_values[blocks] = pmw
        return pmw

    def get_entry(self, query_id, blocks):
        if blocks in self.key_values:
            return self.key_values[blocks]
        return None

    def update_entry(self, query_id, query, blocks, true_result, noise_std, noise):
        pmw = self.get_entry(query_id, blocks)
        if not pmw:  # If there is no PMW for the blocks then create it
            pmw = self.add_entry(blocks)

        # Compute laplace noise from noise_std
        blocks_size = get_blocks_size(blocks, self.config.blocks_metadata)
        sensitivity = 1 / blocks_size
        noisy_result = true_result + noise
        laplace_scale = noise_std / math.sqrt(2)
        laplace_noise = laplace_scale / sensitivity

        # If the new entry is good enough for the PMW
        if pmw.nu >= laplace_noise:
            pmw.external_update(query, noisy_result)

    def estimate_run_budget(self, query_id, query, blocks, noise_std):
        """
        Checks the cache and returns the budget we need to spend if we want to run this query with given accuracy guarantees.
        """

        pmw = self.get_entry(query_id, blocks)
        run_budget = (
            PMW(
                blocks, self.config.blocks_metadata, **self.pmw_args
            ).estimate_run_budget(query)
            if not pmw
            else pmw.estimate_run_budget(query)
        )
        # TODO: This is leaking privacy, assume we have a good estimate already.
        return run_budget
