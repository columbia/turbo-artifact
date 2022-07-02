import random
import collections

from pathlib import Path
REPO_ROOT = Path(__file__).parent.parent
LOGS_PATH = REPO_ROOT.joinpath("exps/exps/logs")
RAY_LOGS = LOGS_PATH

random.seed(99)

class Task:
    def __init__(self, task_id, blocks):
        self.id = task_id
        self.blocks = blocks
        self.result = None
        self.substitutes = {}


    def dump(self,):
        return {
                "id": self.id,
                "result": self.result,
                "blocks": self.blocks,
                "substitutes": {str(key): value for key, value in self.substitutes.items()},
                }


def flatten_list(x):
    if isinstance(x, list):
        return [a for i in x for a in flatten_list(i)]
    else:
        return [x]

def flatten(x):
    if isinstance(x, (list,tuple)):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def workload_pattern(max_queries, max_blocks, query_nums, total_tasks):
    possible_requested_blocks = []
    for i in range(max_blocks):
        for j in range(i, max_blocks):
            possible_requested_blocks += [(i+1, j+1)]
    print("Possible requested blocks", possible_requested_blocks)

    cached_tasks = []
    uncached_tasks = []
    indices = random.sample(range(len(possible_requested_blocks)), total_tasks)
    for i in range(len(possible_requested_blocks)):
        if i in indices:
            cached_tasks += [Task(query_nums[0], possible_requested_blocks[i])]
        else:
            uncached_tasks += [Task(query_nums[0], possible_requested_blocks[i])]

    print("Cached tasks")
    for i in cached_tasks:
        print(i.blocks)
    print("Uncached tasks")
    for i in uncached_tasks:
            print(i.blocks)

    return cached_tasks, uncached_tasks
