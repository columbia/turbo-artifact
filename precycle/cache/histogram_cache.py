import torch
from copy import deepcopy
from precycle.budget.histogram import DenseHistogram, flat_indices


class CacheKey:
    def __init__(self, blocks):
        self.key = blocks


class CacheEntry:
    def __init__(self, histogram, bin_updates, bin_thresholds) -> None:
        self.histogram = histogram
        self.bin_updates = bin_updates
        self.bin_thresholds = bin_thresholds


class MockHistogramCache:
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

    def create_new_entry(self, blocks):

        cache_entry = None
        if self.config.cache.probabilistic_cfg.bootstrapping == True:
            # Bootstrapping: creating a histogram for a new block or node and
            # initializing it with the histogram of the previous block or the children nodes
            (i, j) = blocks
            node_size = j - i + 1
            if node_size == 1 and i > 0:  # leaf node
                # Find the first previous block in cache to initialize from
                for x in reversed(range(i)):
                    cache_entry = self.read_entry((x, x))
                    if cache_entry is not None:
                        break
            else:  # not leaf node - aggregate children
                # Get children nodes
                left_child = (i, i + node_size / 2 - 1)
                right_child = (i + node_size / 2, j)
                left_child_entry = self.read_entry((left_child[0], left_child[1]))
                right_child_entry = self.read_entry((right_child[0], right_child[1]))
                if left_child_entry and right_child_entry:
                    new_histogram = DenseHistogram(self.domain_size)
                    new_histogram.tensor = torch.div(
                        torch.add(
                            left_child_entry.histogram.tensor,
                            right_child_entry.histogram.tensor,
                        ),
                        2,
                    )
                    new_bin_updates = torch.div(
                        torch.add(
                            left_child_entry.bin_updates, right_child_entry.bin_updates
                        ),
                        2,
                    )
                    new_bin_thresholds = torch.div(
                        torch.add(
                            left_child_entry.bin_thresholds,
                            right_child_entry.bin_thresholds,
                        ),
                        2,
                    )
                    cache_entry = CacheEntry(
                        histogram=new_histogram,
                        bin_updates=new_bin_updates,
                        bin_thresholds=new_bin_thresholds,
                    )

        if cache_entry:
            new_cache_entry = CacheEntry(
                histogram=deepcopy(cache_entry.histogram),
                bin_updates=deepcopy(cache_entry.bin_updates),
                bin_thresholds=deepcopy(cache_entry.bin_thresholds),
            )
        else:
            new_cache_entry = CacheEntry(
                histogram=DenseHistogram(self.domain_size),
                bin_updates=torch.zeros(
                    size=(1, self.domain_size), dtype=torch.float64
                ),
                bin_thresholds=torch.ones(
                    size=(1, self.domain_size), dtype=torch.float64
                )
                * self.bin_thershold,
            )
        return new_cache_entry

    def update_entry_histogram(self, query, blocks, noisy_result):
        cache_entry = self.read_entry(blocks)
        if not cache_entry:
            cache_entry = self.create_new_entry(blocks)

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

        bin_updates = [cache_entry.bin_updates[i] for i in flat_indices(query)]
        new_threshold = min(bin_updates) + self.bin_thershold_step
        for i in flat_indices(query):
            cache_entry.bin_thresholds[i] = new_threshold
        # # Write updated entry
        self.write_entry(blocks, cache_entry)

    def get_bin_updates(self, blocks, query):
        cache_entry = self.read_entry(blocks)
        assert cache_entry is not None
        bin_updates = [cache_entry.bin_updates[i] for i in flat_indices(query)]
        return bin_updates

    def get_bin_thresholds(self, blocks, query):
        cache_entry = self.read_entry(blocks)
        assert cache_entry is not None
        bin_thresholds = [cache_entry.bin_thresholds[i] for i in flat_indices(query)]
        return bin_thresholds

    def is_query_hard(self, query, blocks):
        cache_entry = self.read_entry(blocks)
        if not cache_entry:
            return True

        # If each bin has been updated at least <bin-threshold> times the query is easy
        for i in flat_indices(query):
            if cache_entry.bin_updates[i] < cache_entry.bin_thresholds[i]:
                return True

        return False
