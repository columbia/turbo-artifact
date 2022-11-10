from privacypacking.cache.cache import Cache, R, A
from privacypacking.budget.block import HyperBlock
import yaml
import math


class DeterministicCache(Cache):
    def __init__(
        self,
    ):
        self.key_values = {}

    def add_entry(self, query_id, hyperblock_id, result, noise):
        if query_id not in self.key_values:
            self.key_values[query_id] = {}
        if hyperblock_id not in self.key_values[query_id]:
            self.key_values[query_id].update({hyperblock_id: (result, noise)})

    def get_entry(self, query_id, hyperblock_id):
        result = None
        noise = None
        if query_id in self.key_values:
            if hyperblock_id in self.key_values[query_id]:
                (result, noise) = self.key_values[query_id][hyperblock_id]
        return result, noise

    def run(
        self, query_id, query, run_budget, hyperblock: HyperBlock
    ):  # TODO: strip the caches from the 'run' functionality?
        budget = None
        result, noise = self.get_entry(query_id, hyperblock.id)
        if not result:  # If result is not in the cache run fresh and store
            result, noise = hyperblock.run_dp(query, run_budget)
            self.add_entry(
                query_id, hyperblock.id, result, noise
            )  # Add result in cache
            budget = run_budget
        return result, budget, noise

    def dump(self):
        res = yaml.dump(self.key_values)
        print("Results", res)

    # Minimizing aggragations
    # Cost model    # TODO: remove this functionality from the Cache
    def get_cost(self, plan, blocks, structure_constraint=False):
        if isinstance(plan, A):  # Aggregate cost of arguments/operators
            return sum([self.get_cost(x, blocks) for x in plan.l])

        elif isinstance(plan, R):  # Get cost of Run operator
            if structure_constraint and not self.satisfies_constraint(plan.blocks):
                return math.inf

            block_ids = list(range(plan.blocks[0], plan.blocks[-1] + 1))
            hyperblock = HyperBlock({key: blocks[key] for key in block_ids})

            result, _ = self.get_entry(plan.query_id, hyperblock.id)
            if result:
                return 1  # Already cached
            else:
                demand = {key: plan.budget for key in block_ids}
                if not hyperblock.can_run(demand):
                    return math.inf  # This hyperblock does not have enough budget

                return 1  # Even if there is at least a little budget left in the hyperblock we assume the cost is 1

    def satisfies_constraint(self, blocks):
        bf = 2  # Branching factor
        size = blocks[1] - blocks[0] + 1
        if not math.log(size, bf).is_integer():
            return False
        # if size > 1 and not (blocks[1] % bf):
        if (blocks[0] % size) != 0:
            return False
        return True
