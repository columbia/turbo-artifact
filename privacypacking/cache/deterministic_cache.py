from queue import Empty

from privacypacking.cache import Cache, R, C
from privacypacking.cache.utils import get_splits
from termcolor import colored

class DeterministicCache(Cache):
    def __init__(self, max_aggregations_allowed, scheduler):
        self.max_aggregations_allowed = max_aggregations_allowed
        self.scheduler = scheduler
        self.results = {}

    def dump(self,):
        res = yaml.dump(self.results)
        print("Results", res)

    def can_run(self, scheduler, blocks, budget):
        demand = {}
        for block in range(blocks[0], blocks[-1]+1):
            demand[block] = budget
        # Add other constraints too here
        # for block in demand.keys():
            # print(f"             block {block} - available - {scheduler.blocks[block].remaining_budget}")
        return scheduler.can_run(demand)

    def add_result(self, query_id, blocks, budget, result):
        if query_id not in self.results:
            self.results[query_id] = {}
        if blocks not in self.results[query_id]:
            self.results[query_id].update({blocks: (budget.epsilon, result)})

    def run_cache(self, query_id, blocks, budget):
        if query_id in self.results:
            if blocks in self.results[query_id]:
                (_, result) = self.results[query_id][blocks]
                return result

    def get_execution_plan(self, query_id, blocks, budget):

        max_num_aggregations = min(self.max_aggregations_allowed, len(blocks))

        plan = []
        for i in range(max_num_aggregations+1):      # Prioritizing smallest number of aggregations
            splits = get_splits(blocks, i)
            for split in splits:
                # print("split", split)
                
                for x in split:
                    x = (x[0], x[-1])
                    # print("         x", x)

                    if self.run_cache(query_id, x, budget) is not None:
                        plan += [F(query_id, x, budget)]

                    elif self.can_run(self.scheduler, x, budget):
                        plan += [R(query_id, x, budget)]

                    else:
                        plan = []
                        break

                if plan:
                    return A(plan)
        return None
