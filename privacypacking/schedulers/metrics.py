import time
from typing import Dict, List, Type

import numpy as np
from loguru import logger
from scipy.sparse import spmatrix
from scipy.sparse.dok import dok_matrix

from privacypacking.budget import ALPHAS, Block, Task
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
                    demand_fractions.append(
                        1
                        / (
                            demand_budget.epsilon(alpha)
                            / (block_initial_budget.epsilon(alpha) * task.profit)
                        )
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
        # return task.id
        # The smallest id has the highest priority
        return 1 / (task.id + 1)


class FlatRelevance(Metric):
    @staticmethod
    def apply(task: Task, blocks: Dict[int, Block], tasks: List[Task] = None) -> float:
        logger.info(f"Computing FlatRelevance for task {task.id}.")
        cost = 0.0
        for block_id, budget in task.budget_per_block.items():
            for alpha in budget.alphas:
                demand = budget.epsilon(alpha)
                capacity = blocks[block_id].initial_budget.epsilon(alpha)
                logger.info(
                    f"b{block_id}, alpha: {alpha}, demand: {demand}, capacity: {capacity}. Current cost: {cost}"
                )
                if capacity > 0:
                    cost += demand / capacity
        task.cost = cost
        logger.info(f"Task {task.id} cost: {cost} profit: {task.profit / cost} ")
        return task.profit / cost


class DynamicFlatRelevance(Metric):
    @staticmethod
    def apply(task: Task, blocks: Dict[int, Block], tasks: List[Task] = None) -> float:
        logger.info(f"Computing DynamicFlatRelevance for task {task.id}.")
        cost = 0.0
        for block_id, budget in task.budget_per_block.items():
            for alpha in budget.alphas:
                demand = budget.epsilon(alpha)
                remaining_budget = blocks[block_id].budget.epsilon(alpha)
                logger.info(
                    f"b{block_id}, alpha: {alpha}, demand: {demand}, remaining_budget: {remaining_budget}. Current cost: {cost}"
                )
                if remaining_budget > 0:
                    cost += demand / remaining_budget
        task.cost = cost
        logger.info(f"Task {task.id} cost: {cost} profit: {task.profit / cost} ")
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
                if block_id not in overflow_b_a:
                    overflow_b_a[block_id] = {}
                for a in block_demand.alphas:
                    if a not in overflow_b_a[block_id]:
                        overflow_b_a[block_id][a] = -blocks[
                            block_id
                        ].initial_budget.epsilon(a)
                    overflow_b_a[block_id][a] += block_demand.epsilon(a)

        costs = {}
        for block_id_, block_demand_ in task.budget_per_block.items():
            costs[block_id_] = 0
            for alpha in block_demand_.alphas:
                demand = block_demand_.epsilon(alpha)
                overflow = overflow_b_a[block_id_][alpha]
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
            return float("inf")
        return task.profit / total_cost


# TODO: vectorize and cache things to speed up


# class Vectorized


class VectorizedBatchOverflowRelevance(Metric):
    @staticmethod
    def compute_relevance_matrix(
        blocks: Dict[int, Block],
        tasks: List[Task] = None,
        drop_blocks_with_no_contention=False,
    ) -> np.ndarray:
        sum_demands = sum((task.demand_matrix.toarray() for task in tasks))
        logger.info(f"Sum of demands: {sum_demands}")
        # logger.info(f"Task 0 demands: {tasks[0].demand_matrix.toarray()}")

        # Compute the negative available unlocked budget
        n_blocks = len(blocks)
        n_alphas = len(ALPHAS)
        overflow = np.zeros((n_blocks, n_alphas))
        for block_id in range(n_blocks):
            for alpha_index, alpha in enumerate(ALPHAS):
                eps = blocks[block_id].available_unlocked_budget.epsilon(alpha)
                overflow[block_id, alpha_index] = -eps
                # NOTE: we could also accept negative available unlocked budget and drop alphas. Maybe not necessary.
                # if eps >= 0:
                #     overflow[block_id, alpha] = -eps
                # else:
                #     # There is no available budget, so this alpha is not relevant
                #     overflow[block_id, alpha] = float("inf")

        overflow += sum_demands

        relevance = np.reciprocal(overflow)

        if drop_blocks_with_no_contention:
            # If a block has an alpha without contention, the relevance should be 0 because we can allocate everything
            # TODO: do we really need this? It doesn't look neat. I'll deactivate it by default.
            for block_id in range(n_blocks):
                min_overflow = np.min(overflow[block_id, :])
                if min_overflow <= 0:
                    overflow[block_id, :] = np.zeros(n_alphas)

        logger.info(f"Relevance: {relevance}")

        return relevance

    @staticmethod
    def apply(
        task: Task,
        blocks: Dict[int, Block],
        tasks: List[Task] = None,
        relevance_matrix: dict = None,
    ) -> float:
        cost = np.multiply(task.demand_matrix.toarray(), relevance_matrix).sum()
        return task.profit / cost if cost > 0 else float("inf")


class BatchOverflowRelevance(Metric):
    @staticmethod
    def compute_overflow(blocks: Dict[int, Block], tasks: List[Task] = None) -> dict:
        overflow_b_a = {}
        for t in tasks:
            for block_id, block_demand in t.budget_per_block.items():
                if block_id not in overflow_b_a:
                    overflow_b_a[block_id] = {}
                for a in block_demand.alphas:
                    if a not in overflow_b_a[block_id]:
                        # NOTE: This is the only difference with (offline) OverflowRelevance
                        # overflow_b_a[block_id][a] = -blocks[
                        #     block_id
                        # ].initial_budget.epsilon(a)
                        available_unlocked_budget = blocks[
                            block_id
                        ].available_unlocked_budget.epsilon(a)

                        if available_unlocked_budget > 0:
                            overflow_b_a[block_id][a] = -available_unlocked_budget
                            logger.debug(
                                f"b{block_id}, alpha: {a}, available unlocked budget: {blocks[block_id].available_unlocked_budget.epsilon(a)}"
                            )
                        else:
                            # Alphas is consumed at this point
                            overflow_b_a[block_id][a] = float("inf")

                        # NOTE: what should we do with negative available unlocked budget?
                        # We could drop these orders in the computation, but they might become relevant
                        # again if we unlock more budget later.
                        # OverflowRelevance can still penalize a bit tasks that negative orders (high overflow but non-zero relevance)

                    # TODO: this code is horribly inefficient

                    # For exhausted alphas, the overflow remains infinite
                    overflow_b_a[block_id][a] += block_demand.epsilon(a)
        return overflow_b_a

    @staticmethod
    def apply(
        task: Task,
        blocks: Dict[int, Block],
        tasks: List[Task] = None,
        overflow: dict = None,
    ) -> float:
        if overflow:
            logger.info("Using precomputed overflow")
            overflow_b_a = overflow
        else:
            logger.info("Computing fresh overflow")
            overflow_b_a = BatchOverflowRelevance.compute_overflow(blocks, tasks)

        costs = {}
        for block_id_, block_demand_ in task.budget_per_block.items():
            costs[block_id_] = 0
            for alpha in block_demand_.alphas:
                demand = block_demand_.epsilon(alpha)
                overflow = overflow_b_a[block_id_][alpha]
                logger.debug(
                    f"b{block_id_}, alpha: {alpha}, demand: {demand}, overflow: {overflow}. Current cost: {costs[block_id_]}"
                )
                if overflow > 0:
                    costs[block_id_] += demand / overflow
                else:
                    # There is no contention on this block!
                    costs[block_id_] = 0
                    break
        total_cost = 0
        for cost in costs.values():
            total_cost += cost
        task.cost = total_cost
        if total_cost <= 0:
            return float("inf")

        logger.info(
            f"Task {task.id} cost: {total_cost} profit: {task.profit / total_cost} "
        )

        return task.profit / total_cost

    @staticmethod
    def is_dynamic():
        return True
