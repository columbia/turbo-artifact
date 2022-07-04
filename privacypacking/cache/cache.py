import yaml
import itertools
from privacypacking.cache.utils import (
    get_splits,
    get_subsets_of_contiguous_blocks,
    absolute_mean_error,
    have_enough_budget,
)
from termcolor import colored


class Cache:
    def __init__(self, task_lifetime, max_substitutes_allowed):
        self.max_substitutes_allowed = max_substitutes_allowed
        self.task_lifetime = task_lifetime
        self.results = {}
        self.distances = {}
        self.substitute_results = {}
        self.exhausted_blocks = set()

    def dump(self,):
        res = yaml.dump(self.results)
        dis = yaml.dump(self.distances)
        print("Results", res)
        print("Distances", dis)

    def add_result(self, query_type, blocks, result):
        if query_type not in self.results:
            self.results[query_type] = {}
        if blocks not in self.results[query_type]:
            self.results[query_type].update({blocks: result})
            if query_type not in self.distances:
                self.distances[query_type] = {}

    def add_substitute_result(self, query_type, blocks, result):
        if query_type not in self.substitute_results:
            self.substitute_results[query_type] = {}
        if blocks not in self.substitute_results[query_type]:
            self.substitute_results[query_type].update({blocks: result})

    def find_result(self, query_type, blocks):
        if query_type in self.results:
            if blocks in self.results[query_type]:
                return self.results[query_type][blocks]
        if query_type in self.substitute_results:
            if blocks in self.substitute_results[query_type]:
                return self.substitute_results[query_type][blocks]
        return None

    def remove(self, block_id, curr_num_blocks):
        self.exhausted_blocks.add(block_id)
        # Clean from results
        if curr_num_blocks > block_id + self.task_lifetime:
            for k in self.results.keys():
                results = self.results[k]
                delete = [
                    block
                    for block in results.keys()
                    if block[0] <= block_id <= block[-1]
                ]
                for block in delete:
                    del self.results[k][block]

        # Clean from distances
        for k in self.distances.keys():
            distances = self.distances[k]
            for block in distances.keys():
                for v in distances[block][:]:
                    blocks = v[0]
                    if blocks[0] <= block_id <= blocks[-1]:
                        self.distances[k][block].remove(v)

    def compute_distances(self, query_type, blocks):
        task_results = self.results[query_type]
        result = task_results[blocks]

        for blocks_ in task_results.keys():
            if blocks_ != blocks:
                result_ = task_results[blocks_]
                error = absolute_mean_error(result, result_)

                exhausted_block = False
                for b in range(blocks_[0], blocks_[1] + 1):
                    if b in self.exhausted_blocks:
                        exhausted_block = True
                if not exhausted_block:
                    if blocks not in self.distances[query_type]:
                        self.distances[query_type][blocks] = []
                    self.distances[query_type][blocks] += [(blocks_, error)]

                if blocks_ not in self.distances[query_type]:
                    self.distances[query_type][blocks_] = []
                self.distances[query_type][blocks_] += [(blocks, error)]
        # print(colored("Distances distances...", 'yellow'), self.distances)

    def get_substitute_blocks(self, task, sys_blocks):
        direct_substitutes = {}

        def add_to_direct_substitutes(k, val):
            if k not in direct_substitutes:
                direct_substitutes[k] = [val]
            else:
                direct_substitutes[k] += [val]

        bs = sorted(list(task.budget_per_block.keys()))

        # Fill array with direct substitutes that have enough budget
        subsets = get_subsets_of_contiguous_blocks(bs)
        for blocks in subsets:
            if have_enough_budget(blocks, task.budget, sys_blocks):
                add_to_direct_substitutes(blocks, blocks)

            if blocks in self.distances[task.query_type]:
                r = self.distances[task.query_type][blocks]
                for (blocks_, error) in r:
                    #  Substitutes have at most the same cardinality as blocks
                    if blocks[1] - blocks[0] + 1 >= blocks_[1] - blocks_[0] + 1:
                        if error <= task.k and have_enough_budget(
                            blocks_, task.budget, sys_blocks
                        ):
                            add_to_direct_substitutes(blocks, blocks_)

        # Get all substitutes
        num_substitutions = self.max_substitutes_allowed
        substitutes_set = set()
        for i in range(num_substitutions):
            splits = get_splits(bs, i)
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
                        substitutes_set.add(
                            tuple(
                                sorted(
                                    [
                                        x
                                        for s in substitute
                                        for x in range(s[0], s[-1] + 1)
                                    ]
                                )
                            )
                        )
        # Order them according to a heuristic
        for substitute in substitutes_set:
            yield substitute
