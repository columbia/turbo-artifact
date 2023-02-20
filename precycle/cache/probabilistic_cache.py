import torch
from precycle.budget.histogram import DenseHistogram, flat_indices


class ProbabilisticCache:
    def __init__(self, config):
        raise NotImplementedError


class CacheKey:
    def __init__(self, blocks):
        self.key = blocks


class CacheEntry:
    def __init__(self, histogram, bin_updates, bin_thresholds) -> None:
        self.histogram = histogram
        self.bin_updates = bin_updates
        self.bin_thresholds = bin_thresholds


class MockProbabilisticCache:
    def __init__(self, config):
        self.kv_store = {}
        self.config = config
        self.blocks_metadata = self.config.blocks_metadata
        self.learning_rate = config.cache.probabilistic_cfg.learning_rate
        self.domain_size = self.blocks_metadata["domain_size"]
        heuristic = config.cache.probabilistic_cfg.heuristic
        _, heuristic_params = heuristic.split(":")

        threshold, step = heuristic_params.split("-")
        self.bin_thershold = int(threshold)
        self.bin_thershold_step = int(step)

    def write_entry(self, blocks, cache_entry):
        key = CacheKey(blocks).key
        self.kv_store[key] = {
            "histogram": cache_entry.histogram,
            "bin_updates": cache_entry.bin_updates,
            "bin_thresholds": cache_entry.bin_thresholds,
        }

    def read_entry(self, blocks):
        key = CacheKey(blocks).key
        if key in self.kv_store:
            entry = self.kv_store[key]
            return CacheEntry(
                entry["histogram"], entry["bin_updates"], entry["bin_thresholds"]
            )
        return None

    def create_new_entry(self):
        return CacheEntry(
            histogram=DenseHistogram(self.domain_size),
            bin_updates=torch.zeros(size=(1, self.domain_size), dtype=torch.float64),
            bin_thresholds=torch.ones(size=(1, self.domain_size), dtype=torch.float64)
            * self.bin_thershold,
        )

    def update_entry_histogram(self, query, blocks, noisy_result):
        cache_entry = self.read_entry(blocks)
        if not cache_entry:
            cache_entry = self.create_new_entry()

        # Do External Update on the histogram - update bin counts too
        predicted_output = cache_entry.histogram.run(query)

        # Increase weights iff predicted_output is too small
        lr = self.learning_rate / 8
        if noisy_result < predicted_output:
            lr *= -1

        # Multiplicative weights update for the relevant bins
        for i in flat_indices(query):
            cache_entry.histogram.tensor[i] *= torch.exp(query[i] * lr)
            cache_entry.bin_updates[i] += 1
        cache_entry.histogram.normalize()

        # Write updated entry
        self.write_entry(blocks, cache_entry)

    def update_entry_threshold(self, blocks, query):
        cache_entry = self.read_entry(blocks)
        assert cache_entry is not None
        for i in flat_indices(query):
            cache_entry.bin_thresholds[i] += self.bin_thershold_step
        # Write updated entry
        self.write_entry(blocks, cache_entry)

    def is_query_hard(self, query, blocks):
        cache_entry = self.read_entry(blocks)
        if not cache_entry:
            return True

        # If each bin has been upated at least <bin-threshold> times the query is easy
        for i in flat_indices(query):
            if cache_entry.bin_updates[i] < cache_entry.bin_thresholds[i]:
                return True

        return False
