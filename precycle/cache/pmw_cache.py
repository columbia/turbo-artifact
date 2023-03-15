from precycle.cache.pmw import PMW
from precycle.utils.utility_theorems import get_pmw_epsilon, get_sv_epsilon
from precycle.utils.utils import get_blocks_size


class MockPMWCache:
    def __init__(self, config):
        self.key_values = {}
        self.config = config
        self.pmw_alpha = config.alpha
        self.pmw_beta = config.beta
        self.blocks_metadata = config.blocks_metadata

    def add_entry(self, blocks):
        n = get_blocks_size(blocks, self.blocks_metadata)
        # epsilon = get_pmw_epsilon(self.pmw_alpha, self.pmw_beta, n, 1)
        epsilon = get_sv_epsilon(self.pmw_alpha, self.pmw_beta, n)
        pmw = PMW(
            alpha=self.pmw_alpha,
            epsilon=epsilon,
            n=n,
            id=str(blocks)[1:-1].replace(", ", "-"),
            domain_size=self.blocks_metadata["domain_size"],
            config=self.config,
        )
        self.key_values[blocks] = pmw
        return pmw

    def get_entry(self, blocks):
        if blocks in self.key_values:
            return self.key_values[blocks]
        return None
