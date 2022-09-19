from queue import Empty
from privacypacking.cache.utils import (
    get_splits,
)
from termcolor import colored


class R:
    def __init__(self, task_id, blocks, budget) -> None:
        self.task_id = task_id
        self.blocks = blocks

class F:
    def __init__(self, task_id, blocks, budget) -> None:
        self.task_id = task_id
        self.blocks = blocks

class A:
    def __init__(self, l) -> None:
        self.l = l


class Cache:
    def __init__(self, max_aggregations_allowed, disable_dp):
        self.max_aggregations_allowed = max_aggregations_allowed
        self.disable_dp = disable_dp
        self.results = {}

    def dump(self,):
        res = yaml.dump(self.results)
        print("Results", res)

    def can_run(self, scheduler, blocks, budget):
        demand = {}
        for block in range(blocks[0], blocks[-1]+1):
            demand[block] = budget
        # Add other constraints too here
        return scheduler.can_run(demand)

    def add_result(self, query_id, blocks, budget, result):
        if query_id not in self.results:
            self.results[query_id] = {}
        if blocks not in self.results[query_id]:
            self.results[query_id].update({blocks: (budget.epsilon(0.0), result)})
            if query_id not in self.distances:
                self.distances[query_id] = {}

    def find_result(self, query_id, blocks, budget):
        # budget = budget.epsilon(0.0)
        if query_id in self.results:
            if blocks in self.results[query_id]:
                (_, result) = self.results[query_id][blocks]
                return result

    def get_execution_plan(self, query_id, blocks, budget, scheduler):
        if query_id not in self.results:            # Fast way out
            return None

        max_num_aggregations = min(self.max_aggregations_allowed, len(blocks))

        plan = []
        for i in range(max_num_aggregations):      # Prioritizing smallest number of aggregations
            splits = get_splits(blocks, i)
            for split in splits:
                
                for x in split:
                    x = (x[0], x[-1])

                    if self.find_result(x) is not None:
                        plan += self.F(query_id, x, budget)

                    elif self.can_run(scheduler, x, budget):
                        plan += self.R(query_id, x, budget)

                    else:
                        plan = []
                        break

                if plan is not Empty:
                    return self.A(plan)
        return None
