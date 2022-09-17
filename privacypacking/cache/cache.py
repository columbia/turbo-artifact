import yaml
import itertools
from privacypacking.cache.utils import (
    get_splits,
    get_subsets_of_contiguous_blocks,
    upper_bound_normalized_absolute_mean_error,
    normalized_absolute_mean_error,
)
from termcolor import colored
import time
import collections
import numpy as np


class Cache:
    def __init__(self, max_substitutes_allowed, disable_dp):
        self.max_substitutes_allowed = max_substitutes_allowed
        self.disable_dp = disable_dp
        self.results = {}
        self.distances = {}
        self.substitute_results = {}
        self.exhausted_blocks = set()
        self.blocks_to_be_deleted = set()

    def dump(self,):
        res = yaml.dump(self.results)
        dis = yaml.dump(self.distances)
        print("Results", res)
        print("Distances", dis)

    def add_result(self, query_id, blocks, epsilon, result):
        if query_id not in self.results:
            self.results[query_id] = {}
        if blocks not in self.results[query_id]:
            self.results[query_id].update({blocks: (epsilon, result)})
            if query_id not in self.distances:
                self.distances[query_id] = {}

    def add_substitute_result(self, query_id, blocks, epsilon, result):
        if query_id not in self.substitute_results:
            self.substitute_results[query_id] = {}
        if blocks not in self.substitute_results[query_id]:
            self.substitute_results[query_id].update({blocks: (epsilon, result)})

    def find_result(self, query_id, blocks):
        if query_id in self.results:
            if blocks in self.results[query_id]:
                (_, result) = self.results[query_id][blocks]
                return result

    def find_substitute_result(self, query_id, blocks):
        if query_id in self.substitute_results:
            if blocks in self.substitute_results[query_id]:
                (_, result) = self.substitute_results[query_id][blocks]
                return result

    def remove(self, block_id, curr_num_blocks):
        self.blocks_to_be_deleted.add(block_id)
        grace_period = 0
        # self.exhausted_blocks.add(block_id)
        delete = []
        for b in self.blocks_to_be_deleted:
            if curr_num_blocks > b + grace_period:
                self.exhausted_blocks.add(b)
                delete.append(b)

                for k in self.distances.keys():
                    distances = self.distances[k]
                    for block in distances.keys():
                        for v in distances[block][:]:
                            blocks = v[0]
                            if blocks[0] <= b <= blocks[-1]:
                                self.distances[k][block].remove(v)
        for d in delete:
            self.blocks_to_be_deleted.remove(d)

    def compute_distances(self, query_id, blocks, curr_num_blocks, k):
        task_results = self.results[query_id]
        result = task_results[blocks]

        for blocks_ in task_results.keys():
            if blocks_ != blocks:
                result_ = task_results[blocks_]
                if self.disable_dp:
                    error = normalized_absolute_mean_error(result, result_)
                else:
                    error = upper_bound_normalized_absolute_mean_error(result, result_)
                if error <= k:
                    if not self.is_exhausted(blocks_):
                        if blocks not in self.distances[query_id]:
                            self.distances[query_id][blocks] = []
                        self.distances[query_id][blocks] += [(blocks_, error)]

                    # For backwards substitution
                    # if (blocks_[1]+1) + 100 > curr_num_blocks:
                    if blocks_ not in self.distances[query_id]:
                        self.distances[query_id][blocks_] = []
                    self.distances[query_id][blocks_] += [(blocks, error)]
                # print(colored("Distances distances...", 'yellow'), self.distances)

    def is_exhausted(self, blocks):
        for b in range(blocks[0], blocks[1] + 1):
            if b in self.exhausted_blocks:
                return True
        return False

    def get_substitute_blocks(self, query_id, query_type, original_blocks, k):
        if query_id not in self.results or query_id not in self.distances:
            return tuple(original_blocks)

        direct_substitutes = {}

        def add_to_direct_substitutes(k, val):
            if k not in direct_substitutes:
                direct_substitutes[k] = [val]
            else:
                direct_substitutes[k] += [val]

        # Fill array with direct substitutes that have enough budget
        subsets = get_subsets_of_contiguous_blocks(original_blocks)
        for blocks in subsets:
            if not self.is_exhausted(blocks):
                add_to_direct_substitutes(blocks, list(range(blocks[0], blocks[-1]+1)))

            if blocks in self.distances[query_id]:
                blockslen = blocks[1]-blocks[0]+1
                r = self.distances[query_id][blocks]
                for (blocks_, error) in r:
                    blockslen_ = blocks_[1]-blocks_[0]+1
                    #  Substitutes have at most the same cardinality as blocks
                    if blockslen - blockslen_ >= 0:
                        if error <= k and not self.is_exhausted(blocks_):
                            blocks_ = list(range(blocks_[0], blocks_[-1]+1))
                            if query_type != "average":
                                add_to_direct_substitutes(blocks, blocks_)
                            else:
                                q, mod = divmod(blockslen, blockslen_)
                                # print("qmod", q, mod, blocks, blocks_)
                                if not mod and q >= 1:
                                    # print("\nmod", mod, "blocks", blocks, "blocks_", blocks_, list(itertools.chain(*[[blocks_[i]]*q for i in range(blockslen_)])))
                                    add_to_direct_substitutes(blocks, list(itertools.chain(*[[blocks_[i]]*q
                                                                                        for i in range(blockslen_)])))

        # Get all substitutes
        num_substitutions = self.max_substitutes_allowed
        substitutes_set = set()
        for i in range(num_substitutions):
            splits = get_splits(original_blocks, i)
            for split in splits:
                substitutes = []
                for s in split:
                    s = (s[0], s[-1])
                    if s in direct_substitutes:
                        substitutes += [direct_substitutes[s]]
                    else:
                        substitutes = []
                        break
                if substitutes:
                    for substitute in itertools.product(*substitutes):
                        # print("substitute", substitute)
                        sub = tuple(sorted(list(itertools.chain(*substitute))))
                        if self.find_substitute_result(query_id, sub) is not None:
                            yield sub
                        substitutes_set.add(sub)

        def sortfunc(x):
            return [count for _, count in collections.Counter(x).items()]

        for substitute in sorted(substitutes_set, key=sortfunc, reverse=True):
            yield substitute

