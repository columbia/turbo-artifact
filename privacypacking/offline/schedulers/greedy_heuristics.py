import random
from typing import List, Tuple

import numpy as np

from privacypacking.budget import Block, Task
from privacypacking.offline.schedulers.scheduler import Scheduler
from privacypacking.utils.utils import get_block_by_block_id


# TODO: double check block mutability, should be a method?
def greedy_allocation(sorted_tasks: List[Task], blocks: List[Block]) -> List[bool]:
    """Allocate tasks in order until there is no budget left

    Args:
        sorted_tasks (List[Task]): ordered tasks
        blocks (List[Block]): blocks requested by the tasks (will be modified)

    Returns:
        List[bool]: i is True iff task i can be allocated
    """
    n_tasks = len(sorted_tasks)
    allocation = [False] * n_tasks
    for i, task in enumerate(sorted_tasks):
        for block_id in task.block_ids:
            block = get_block_by_block_id(blocks, block_id)
            block_budget = block.budget
            demand_budget = task.budget_per_block[block_id]
            if block_budget >= demand_budget:
                allocation[i] = True
                block.budget = block_budget - demand_budget
    return allocation


# TODO: reverse order + backfill (dual heuristic)


def dominant_shares(task: Task, blocks: List[Block]) -> List[float]:
    demand_fractions = []
    for block_id in task.block_ids:
        block = get_block_by_block_id(blocks, block_id)
        block_initial_budget = block.initial_budget
        demand_budget = task.budget_per_block[block_id]

        # Compute the demand share for each alpha of the block
        for alpha in block_initial_budget.alphas:
            demand_fractions.append(
                demand_budget.epsilon(alpha) / block_initial_budget.epsilon(alpha)
            )

    # Order by highest demand fraction first
    demand_fractions.sort(reverse=True)
    return demand_fractions


# TODO: subclasses
# static relevance values only for now

# TODO: add profit field to the tasks
class GreedyHeuristic(Scheduler):
    def __init__(self, tasks, blocks):
        super().__init__(tasks, blocks)

    def order(self) -> Tuple[List[int], List[Task]]:
        """Sorts the tasks according to a heuristic.
            (Can be stateful in the online case with dynamic heuristics)

        Returns:
            Tuple[List[int], List[Task]]:
            - `sorted_indices` such that `sorted_tasks[i] = tasks[sorted_indices[i]]`
            - A sorted copy of the task list (highest priority first)
        """
        # The default heuristic doesn't do anything
        return list(range(len(self.tasks))), self.tasks.copy()

    def schedule(self):
        n_tasks = len(self.tasks)

        # Sort and allocate in order
        sorted_indices, sorted_tasks = self.order()
        sorted_allocation = greedy_allocation(sorted_tasks, self.blocks)

        # Reindex according to self.tasks
        allocation = [False] * n_tasks
        for i in range(n_tasks):
            allocation[sorted_indices[i]] = sorted_allocation[i]
        return allocation


class OfflineDPF(GreedyHeuristic):
    def order(self) -> Tuple[List[int], List[Task]]:
        """Sorts the tasks by dominant share"""

        n_tasks = len(self.tasks)

        def index_key(index):
            # Lexicographic order (the dominant share is the first component)
            return dominant_shares(self.task[index], self.blocks)

        # Task number i is high priority if it has small dominant share
        original_indices = list(range(n_tasks))
        sorted_indices = original_indices.sorted(key=index_key)
        sorted_tasks = [None] * n_tasks
        for i in range(n_tasks):
            # TODO: copy?
            sorted_tasks[i] = self.tasks[sorted_indices[i]]

        return sorted_indices, sorted_tasks


class FlatRelevance(GreedyHeuristic):
    def order(self) -> Tuple[List[int], List[Task]]:
        """The cost of a task is the sum of its normalized demands"""
        n_tasks = len(self.tasks)

        def index_key(index):
            cost = 0
            task = self.tasks[index]
            for block_id, budget in task.budget_per_block.items():
                for alpha in budget.alphas:
                    demand = budget.epsilon(alpha)
                    capacity = self.blocks[block_id].initial_budget.epsilon(alpha)
                    cost += demand / capacity
            return cost

        original_indices = list(range(n_tasks))
        sorted_indices = original_indices.sorted(key=index_key)
        sorted_tasks = [None] * n_tasks
        for i in range(n_tasks):
            # TODO: copy?
            sorted_tasks[i] = self.tasks[sorted_indices[i]]

        return sorted_indices, sorted_tasks


def main():
    # num_blocks = 1 # single-block case
    num_blocks = 2  # multi-block case

    blocks = [Block.from_epsilon_delta(i, 10, 0.001) for i in range(num_blocks)]
    tasks = (
        [
            create_gaussian_task(i, num_blocks, range(num_blocks), s)
            for i, s in enumerate(np.linspace(0.1, 1, 10))
        ]
        + [
            create_gaussian_task(i, num_blocks, range(num_blocks), l)
            for i, l in enumerate(np.linspace(0.1, 10, 5))
        ]
        + [
            create_subsamplegaussian_task(
                i, num_blocks, range(num_blocks), ds=60_000, bs=64, epochs=10, s=s
            )
            for i, s in enumerate(np.linspace(1, 10, 5))
        ]
    )

    random.shuffle(tasks)
    scheduler = GreedyHeuristic(tasks, blocks)
    allocation = scheduler.schedule()
    scheduler.plot(allocation)


if __name__ == "__main__":
    main()
