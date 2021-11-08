from typing import Dict, List, Type

from privacypacking.budget import Block, Task
from privacypacking.schedulers.scheduler import TaskQueue


class MetricException(Exception):
    pass


class Metric:
    @staticmethod
    def from_str(metric: str) -> Type["Metric"]:
        if metric in globals():
            return globals()[metric]
        else:
            raise MetricException(f"Unknown metric: {metric}")

    @staticmethod
    def apply(queue: TaskQueue, efficiency: float):
        pass

    @staticmethod
    def is_dynamic():
        return False


class DominantShares(Metric):
    @staticmethod
    def apply(
        task: Task, blocks: Dict[int, Block], tasks: List[Task] = None
    ) -> List[float]:
        demand_fractions = []
        for block_id, demand_budget in task.budget_per_block.items():
            block = blocks[block_id]
            block_initial_budget = block.initial_budget
            # Compute the demand share for each alpha of the block
            for alpha in block_initial_budget.alphas:
                # Drop RDP orders that are already negative
                if block_initial_budget.epsilon(alpha) > 0:
                    demand_fractions.append(1 /
                                            (demand_budget.epsilon(alpha)
                        / (block_initial_budget.epsilon(alpha) * task.profit))
                    )
        # Order by highest demand fraction first
        demand_fractions.sort()
        return demand_fractions


# class AvailableDominantShares(Metric):
#     @staticmethod
#     def apply(
#         task: Task, blocks: Dict[int, Block], tasks: List[Task] = None
#     ) -> List[float]:
#         demand_fractions = []
#         for block_id, demand_budget in task.budget_per_block.items():
#             block = blocks[block_id]
#             block_remaining_budget = block.budget
#             # Compute the demand share for each alpha of the block
#             for alpha in block_remaining_budget.alphas:
#                 # Drop RDP orders that are already negative
#                 if block_remaining_budget.epsilon(alpha) > 0:
#                     demand_fractions.append(
#                         demand_budget.epsilon(alpha)
#                         / block_remaining_budget.epsilon(alpha)
#                     )
#         # Order by highest demand fraction first
#         demand_fractions.sort(reverse=True)
#         return demand_fractions


class Fcfs(Metric):
    @staticmethod
    def apply(
        task: Task, blocks: Dict[int, Block] = None, tasks: List[Task] = None
    ) -> id:
        return task.id


class FlatRelevance(Metric):
    @staticmethod
    def apply(task: Task, blocks: Dict[int, Block], tasks: List[Task] = None) -> float:
        cost = 0.0
        for block_id, budget in task.budget_per_block.items():
            for alpha in budget.alphas:
                demand = budget.epsilon(alpha)
                capacity = blocks[block_id].initial_budget.epsilon(alpha)
                if capacity > 0:
                    cost += demand / capacity
        task.cost = cost
        return task.profit / cost


class DynamicFlatRelevance(Metric):
    @staticmethod
    def apply(task: Task, blocks: Dict[int, Block], tasks: List[Task] = None) -> float:
        cost = 0.0
        for block_id, budget in task.budget_per_block.items():
            for alpha in budget.alphas:
                demand = budget.epsilon(alpha)
                remaining_budget = blocks[block_id].budget.epsilon(alpha)
                if remaining_budget > 0:
                    cost += demand / remaining_budget
        task.cost = cost
        return task.profit / cost

    @staticmethod
    def is_dynamic():
        return True


class SquaredDynamicFlatRelevance(Metric):
    @staticmethod
    def apply(task: Task, blocks: Dict[int, Block], tasks: List[Task] = None) -> float:
        total_cost = 0.0
        block_cost = 0
        for block_id, budget in task.budget_per_block.items():
            for alpha in budget.alphas:
                demand = budget.epsilon(alpha)
                remaining_budget = blocks[block_id].budget.epsilon(alpha)
                if remaining_budget > 0:
                    block_cost += demand / remaining_budget
            total_cost += block_cost ** 2
        task.cost = total_cost
        return task.profit / total_cost

    @staticmethod
    def is_dynamic():
        return True


class RoundRobins(Metric):
    @staticmethod
    def apply(task: Task, blocks: Dict[int, Block], tasks: List[Task] = None) -> float:
        pass


class OverflowRelevance(Metric):
    @staticmethod
    def apply(task: Task, blocks: Dict[int, Block], tasks: List[Task] = None) -> float:
        overflow_b_a = {}
        for t in tasks:
            for block_id, block_demand in t.budget_per_block.items():
                for a in block_demand.alphas:
                    if block_id not in overflow_b_a:
                        overflow_b_a[block_id] = {}
                    if a not in overflow_b_a[block_id]:
                        overflow_b_a[block_id][a] = -blocks[
                            block_id
                        ].initial_budget.epsilon(a)
                    overflow_b_a[block_id][a] += block_demand.epsilon(a)

        # print(overflow_b_a)
        costs = {}
        for block_id_, block_demand_ in task.budget_per_block.items():
            costs[block_id_] = 0
            for alpha in block_demand_.alphas:
                demand = block_demand_.epsilon(alpha)
                # print(f"demand: {demand}")
                overflow = overflow_b_a[block_id_][alpha]
                # print(f"cost +=: {demand/ overflow}")
                if overflow > 0:
                    costs[block_id_] += demand / overflow
                else:
                    costs[block_id_] = 0
                    break
        total_cost = 0
        for cost in costs.values():
            total_cost += cost
        task.cost = total_cost
        if total_cost <= 0:
            return float('inf')
        return task.profit / total_cost
