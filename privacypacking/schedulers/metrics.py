from typing import Dict, List, Tuple

from privacypacking.budget import (
    Block,
    Task,
)


def dominant_shares(tasks: List[Task], blocks: Dict[int, Block]) -> List[Task]:
    def task_key(task):
        demand_fractions = []
        for block_id, demand_budget in task.budget_per_block.items():
            block = blocks[block_id]
            block_initial_budget = block.initial_budget
            # Compute the demand share for each alpha of the block
            for alpha in block_initial_budget.alphas:
                # Drop RDP orders that are already negative
                if block_initial_budget.epsilon(alpha) > 0:
                    demand_fractions.append(
                        demand_budget.epsilon(alpha) / block_initial_budget.epsilon(alpha)
                    )
        # Order by highest demand fraction first
        demand_fractions.sort(reverse=True)
        return demand_fractions

    return sorted(tasks, key=task_key)


def fcfs(tasks: List[Task], blocks: Dict[int, Block]) -> List[Task]:
    def task_key(task):
        return task.id

    return sorted(tasks, key=task_key)


def flat_relevance(tasks: List[Task], blocks: Dict[int, Block]) -> List[Task]:
    def task_key(task):
        cost = 0
        for block_id, budget in task.budget_per_block.items():
            for alpha in budget.alphas:
                demand = budget.epsilon(alpha)
                capacity = blocks[block_id].initial_budget.epsilon(alpha)
                if capacity > 0:
                    cost += demand / capacity
        return cost

    return sorted(tasks, key=task_key)


def overflow_relevance(tasks: List[Task], blocks: Dict[int, Block]) -> List[Task]:
    overflow_b_a = {}
    for task in tasks:
        for block_id, block_demand in task.budget_per_block.items():
            for a in block_demand.alphas:
                if block_id not in overflow_b_a:
                    overflow_b_a[block_id] = {}
                if a not in overflow_b_a[block_id]:
                    overflow_b_a[block_id][a] = -blocks[
                        block_id
                    ].initial_budget.epsilon(a)
                overflow_b_a[block_id][a] += block_demand.epsilon(a)

    def task_key(task):
        cost = 0
        for block_id_, block_demand_ in task.budget_per_block.items():
            for alpha in block_demand_.alphas:
                demand = block_demand_.epsilon(alpha)
                overflow = overflow_b_a[block_id_][alpha]
                cost += demand / overflow
                if overflow < 0:
                    cost = 0
        return cost

    return sorted(tasks, key=task_key)


def round_robins(tasks: List[Task], blocks: Dict[int, Block]) -> List[Task]:
    pass
