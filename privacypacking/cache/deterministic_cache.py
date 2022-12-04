from privacypacking.cache.cache import Cache, R, A
from privacypacking.budget.block import HyperBlock
import yaml
import math


class DeterministicCache(Cache):
    def __init__(
        self,
    ):
        self.key_values = {}

    def add_entry(self, query_id, hyperblock_id, result, budget):
        if query_id not in self.key_values:
            self.key_values[query_id] = {}
        # if hyperblock_id not in self.key_values[query_id]:
        self.key_values[query_id].update({hyperblock_id: (result, budget)})

    def get_entry_with_budget(self, query_id, hyperblock_id, budget):
        if query_id in self.key_values:
            if hyperblock_id in self.key_values[query_id]:
                (result, old_budget) = self.key_values[query_id][hyperblock_id]
                if old_budget >= budget:
                    return result, old_budget
        return None, None

    def get_entry(self, query_id, hyperblock_id):
        if query_id in self.key_values:
            if hyperblock_id in self.key_values[query_id]:
                (result, budget) = self.key_values[query_id][hyperblock_id]
                return result, budget
        return None, None

    def run(
        self, query_id, query, run_budget, hyperblock: HyperBlock
    ):  # TODO: strip the caches from the 'run' functionality?
        budget = None
        result, _ = self.get_entry_with_budget(query_id, hyperblock.id, run_budget)
        if not result:  # If result is not in the cache run fresh and store
            result = hyperblock.run_dp(query, run_budget)
            self.add_entry(
                query_id, hyperblock.id, result, run_budget
            )  # Add result in cache
            budget = run_budget
        return result, budget

    def dump(self):
        res = yaml.dump(self.key_values)
        print("Results", res)

    # Minimizing aggragations
    # Cost model    # TODO: remove this functionality from the Cache
    def get_cost(self, plan, blocks, structure_constraint=False, branching_factor=2):
        if isinstance(plan, A):  # Aggregate cost of arguments/operators
            return sum([self.get_cost(x, blocks) for x in plan.l])

        elif isinstance(plan, R):  # Get cost of Run operator
            # print(f"getting cost for {plan}")

            if structure_constraint and not self.satisfies_constraint(
                plan.blocks, branching_factor
            ):
                # print("Cost: Not binary!")
                return math.inf

            block_ids = list(range(plan.blocks[0], plan.blocks[-1] + 1))
            hyperblock = HyperBlock({key: blocks[key] for key in block_ids})

            result, _ = self.get_entry_with_budget(
                plan.query_id, hyperblock.id, plan.budget
            )
            if result:
                return 1  # Already cached
            else:
                demand = {key: plan.budget for key in block_ids}
                if not hyperblock.can_run(demand):
                    # print("Cost: Not enough budget!")
                    return math.inf  # This hyperblock does not have enough budget

                return 1  # Even if there is at least a little budget left in the hyperblock we assume the cost is 1

    # Minimizing Budget
    # Cost model
    def get_entry_budget(self, query_id, blocks):
        result, budget = self.get_entry(query_id, blocks)
        if result:
            return budget.epsilon
        return 0.0

    def satisfies_constraint(self, blocks, branching_factor):
        size = blocks[1] - blocks[0] + 1
        if not math.log(size, branching_factor).is_integer():
            return False
        # if size > 1 and not (blocks[1] % bf):
        if (blocks[0] % size) != 0:
            return False
        return True
