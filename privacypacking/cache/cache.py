import yaml
from itertools import permutations
import numpy as np
from termcolor import colored
from privacypacking.harvestpd.cache.utils import flatten, flatten_list


class Cache:
    def __init__(self, ):
        self.results = {}
        self.distances = {}
        self.substitute_results = {}
        self.exhausted_blocks = set()

    def dump(self, ):
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
            # print(colored("\nComputing new distances...", 'yellow'))
            # self.compute_distances(query_type, blocks)
    #             print(self.dump())

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
        if curr_num_blocks > block_id + 14:
            for k in self.results.keys():
                results = self.results[k]
                delete = [block for block in results.keys() if block[0] <= block_id <= block[-1]]
                for block in delete:
                    del self.results[k][block]

        # Clean from distances
        for k in self.distances.keys():
            # delete = []
            distances = self.distances[k]
            for block in distances.keys():
                # if block_id >= block[0] or block_id <= block[-1]:
                #     delete.append(block)
                for v in distances[block][:]:
                    blocks = v[0]
                    # print("del", blocks)
                    if blocks[0] <= block_id <= blocks[-1]:
                        self.distances[k][block].remove(v)
            # for block in delete:
            #     del self.distances[k][block]

    def compute_distances(self, query_type, blocks):
        task_results = self.results[query_type]
        result = task_results[blocks]

        for blocks_ in task_results.keys():
            if blocks_ != blocks:
                result_ = task_results[blocks_]
                error = self.absolute_mean_error(result, result_)

                exhausted_block = False
                for b in range(blocks_[0], blocks_[1]+1):
                    if b in self.exhausted_blocks:
                        exhausted_block = True
                if not exhausted_block:
                    if blocks not in self.distances[query_type]:
                        self.distances[query_type][blocks] = []
                    self.distances[query_type][blocks] += [(blocks_, error)]

                if blocks_ not in self.distances[query_type]:
                    self.distances[query_type][blocks_] = []
                self.distances[query_type][blocks_] += [(blocks, error)]
        # print(self.distances[4])
        # print(colored("Distances distances...", 'yellow'), self.distances)

    def absolute_mean_error(self, res1, res2):
        return np.abs((res1.to_numpy() - res2.to_numpy())).mean()

    def get_substitute_blocks(self, task, sys_blocks):
        if task.query_type not in self.results:
            return ()

        arr = {}
        bs = sorted(list(task.budget_per_block.keys()))
        blocks = (bs[0], bs[-1])
        substitute_blocks = self.get_substitute_blocks_rec(task.query_type, blocks, task.k,
                                                           task.budget, sys_blocks, arr)
        # print(substitute_blocks)
        subs = set()
        for blocks in substitute_blocks:
            blocks = flatten_list(blocks)
            for i, v in enumerate(blocks):
                blocks[i] = tuple(range(v[0], v[1]+1))
            subs.add(tuple(sorted(flatten(blocks))))
        # for i in subs:
        #     print(i)
        # print(self.distances[4])
        # exit(0)
        return subs

    def get_substitute_blocks_rec(self, query_type, blocks, k, demand, sys_blocks, arr):
        """ returns a list of lists of substitute blocks """
        if blocks in arr:
            return arr[blocks]

        substitutes = []

        if have_enough_budget(blocks, demand, sys_blocks):
            substitutes += [blocks]

        if blocks in self.distances[query_type]:
            r = self.distances[query_type][blocks]
            for (blocks_, error) in r:
                #  Substitutes have at most the same cardinality as blocks
                if blocks[1] - blocks[0] >= blocks_[1] - blocks_[0]:
                    if error <= k and have_enough_budget(blocks_, demand, sys_blocks):
                        substitutes += [blocks_]

        if blocks[1] - blocks[0] > 0:
            for combination in get_combinations(blocks):
                substitute = []
                # print("combination", combination)
                for b in combination:
                    if b != blocks:
                        # print("     get subs of", b)
                        sub = self.get_substitute_blocks_rec(query_type, b, k, demand, sys_blocks, arr)
                        if not sub:
                            print("\n\n\nHAPPENED...\n\n\n")
                            substitute = []
                            break
                        substitute = cross_product(substitute, sub)

                substitutes += substitute
        arr[blocks] = substitutes
        # print(f"     substitutes of {blocks} are {substitutes} ")
        return substitutes


def have_enough_budget(blocks, demand, sys_blocks):
    for b in range(blocks[0], blocks[1]+1):
        if not (sys_blocks[b].remaining_budget-demand).is_positive():
            return False
    return True
    # if b in self.exhausted_blocks:


def cross_product(l1, l2):
    ll = []
    for i in l2:
        if len(l1) == 0:
            ll += [i]
        for j in l1:
            ll += [[j] + [i]]
    return ll


def get_combinations(blocks):
    blocks = list(range(blocks[0], blocks[1] + 1))
    combinations = set()
    size = len(blocks)
    for i in reversed(range(1, size + 1)):
        # list of size i that sum up to size
        div, mod = divmod(size, i)
        ll = [div + 1] * mod + [div] * (i - mod)
        perms = permutations(ll)
        combinations.update(list(perms))
    combs = []
    for comb in combinations:
        x = 0
        c = []
        for v in comb:
            c.append((blocks[x], blocks[x + v - 1]))
            x += v
        combs += [c]
    return combs
