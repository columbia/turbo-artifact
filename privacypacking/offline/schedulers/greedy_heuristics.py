import random
from typing import Dict, List, Tuple

import numpy as np

from privacypacking.budget import Block, Task
from privacypacking.budget.task import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
    UniformTask,
)
from privacypacking.offline.schedulers.scheduler import Scheduler
from privacypacking.utils.scheduling import dominant_shares


# TODO: double check block mutability, should be a method?
def greedy_allocation(sorted_tasks: List[Task], blocks: Dict[int, Block]) -> List[bool]:
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
        for block_id, demand_budget in task.budget_per_block.items():
            block = blocks[block_id]
            if block.budget >= demand_budget:
                allocation[i] = True
                block.budget -= demand_budget
    return allocation


# TODO: reverse order + backfill (dual heuristic)


# TODO: subclasses
# static relevance values only for now

# TODO: add profit field to the tasks
class GreedyHeuristic(Scheduler):
    def __init__(self, tasks, blocks, config=None):
        super().__init__(tasks, blocks, config)

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
            return dominant_shares(self.tasks[index], self.blocks)

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


class OverflowRelevance(GreedyHeuristic):
    # TODO: add preprocessing to rule out the contention.
    def order(self) -> Tuple[List[int], List[Task]]:
        """The dimensions that are closer to being bottlenecks are more relevant
            r_{jk\alpha} = 1/(\sum_{i}w_{ik\alpha} - c_{k\alpha})
        This heuristic only works in the offline case with contention.
        """
        n_tasks = len(self.tasks)

        # overflow_b_α[block_id][α] = \sum_{i}w_{i, block_id, α} - c_{block_id, α}
        overflow_b_α = {}
        for task in self.tasks:
            for block_id, block_demand in task.budget_per_block.items():
                for α in block_demand.alphas:
                    if block_id not in overflow_b_α:
                        overflow_b_α[block_id] = {}
                    if α not in overflow_b_α[block_id]:
                        overflow_b_α[block_id][α] = -self.blocks[
                            block_id
                        ].initial_budget.epsilon(α)
                    overflow_b_α[block_id][α] += block_demand.epsilon(α)

        def index_key(index):
            cost = 0
            task = self.tasks[index]
            for block_id, block_demand in task.budget_per_block.items():
                for alpha in block_demand.alphas:
                    demand = block_demand.epsilon(alpha)
                    overflow = overflow_b_α[block_id][alpha]
                    cost += demand / overflow
                    if overflow < 0:
                        raise Exception("There is no contention for this block")
            return cost

        # TODO: refactor to avoid copying for relevance heuristics?
        # TODO: threshold, exponent on the overflow
        original_indices = list(range(n_tasks))
        sorted_indices = original_indices.sorted(key=index_key)
        sorted_tasks = [None] * n_tasks
        for i in range(n_tasks):
            # TODO: copy?
            sorted_tasks[i] = self.tasks[sorted_indices[i]]

        return sorted_indices, sorted_tasks


# TODO: test and evaluate these greedy heuristics when we have more tooling


def main():
    # num_blocks = 1 # single-block case
    num_blocks = 2  # multi-block case

    blocks = {}
    for i in range(num_blocks):
        blocks[i] = Block.from_epsilon_delta(i, 10, 0.001)

    tasks = (
        [
            UniformTask(
                id=i, profit=1, block_ids=range(num_blocks), budget=GaussianCurve(s)
            )
            for i, s in enumerate(np.linspace(0.1, 1, 10))
        ]
        + [
            UniformTask(
                id=i,
                profit=1,
                block_ids=range(num_blocks),
                budget=LaplaceCurve(l),
            )
            for i, l in enumerate(np.linspace(0.1, 10, 5))
        ]
        + [
            UniformTask(
                id=i,
                profit=1,
                block_ids=range(num_blocks),
                budget=SubsampledGaussianCurve.from_training_parameters(
                    60_000, 64, 10, s
                ),
            )
            for i, s in enumerate(np.linspace(1, 10, 5))
        ]
    )

    random.shuffle(tasks)
    scheduler = GreedyHeuristic(tasks, blocks)
    allocation = scheduler.schedule()
    # config.plotter.plot(tasks, blocks, allocation)


if __name__ == "__main__":
    main()
