from typing import List, Tuple
from privacypacking.budget import Block, Task, ZeroCurve
from privacypacking.utils.utils import get_block_by_block_id
from privacypacking.online.schedulers.scheduler import Scheduler


class DPFBlock:
    """
    A wrapper for the traditional-block.
    It holds a reference to the traditional-block but has some additional DPF properties
    """

    def __init__(self, id, block):
        self.id = id
        self.budget = ZeroCurve()  # Will be gradually unlocking budget till we reach full capacity
        self.allocated_budget = ZeroCurve()  # Budget currently allocated by tasks
        self.block = block

    def unlock_budget(self, fair_share):
        self.budget.increase_budget_by_constant(fair_share, self.block.initial_budget)

    def allocate_budget(self, demand_budget):
        available_budget = self.budget - self.allocated_budget
        diff = available_budget - demand_budget
        max_order = max(diff.epsilons)
        if max_order >= 0:
            self.allocated_budget += demand_budget
            self.block.budget -= demand_budget  # Update the traditional block
            return True
        return False


class DPF(Scheduler):
    # Static variable ; for each traditional block the scheduler creates and holds a corresponding dpf_block
    # that has additional information related to the DPF scheduler
    dpf_blocks = []

    def __init__(self, tasks, blocks, config=None):
        super().__init__(tasks, blocks, config)
        assert config is not None

    def update_dpf_blocks(self):
        pass

    def unlock_block_budgets(self):
        N = self.config.scheduler_N
        new_task = self.tasks[-1]
        for block_id in new_task.block_ids:
            dpf_block = get_block_by_block_id(DPF.dpf_blocks, block_id)
            fair_share = dpf_block.initial_budget / N
            dpf_block.unlock_budget(fair_share)

    def task_dominant_shares(self, task_index: int) -> List[float]:
        demand_fractions = []
        task = self.tasks[task_index]
        for block_id, demand_budget in task.budget_per_block:
            block = get_block_by_block_id(self.blocks, block_id)
            block_initial_budget = block.initial_budget

            # Compute the demand share for each alpha of the block
            for alpha in block_initial_budget.alphas:
                demand_fractions.append(
                    demand_budget.epsilon(alpha) / block_initial_budget.epsilon(alpha)
                )

        # Order by highest demand fraction first
        demand_fractions.sort(reverse=True)
        return demand_fractions

    def order(self) -> List[int]:
        """Sorts the tasks by dominant share"""

        n_tasks = len(self.tasks)

        def index_key(index):
            # Lexicographic order (the dominant share is the first component)
            return self.task_dominant_shares(index)

        # Task number i is high priority if it has small dominant share
        original_indices = list(range(n_tasks))
        sorted_indices = sorted(original_indices, key=index_key, reverse=True)
        return sorted_indices

    def schedule(self):
        n_tasks = len(self.tasks)
        allocation = [False] * n_tasks

        # Update dpf_blocks in case new blocks arrived
        self.update_dpf_blocks()

        # Unlock budgets
        self.unlock_block_budgets()

        # Task indices sorted by smallest dominant share
        sorted_indices = self.order()

        # Try and schedule tasks
        for i in sorted_indices:
            task = self.tasks[i]
            for block_id, demand_budget in task.budget_per_block:
                block = get_block_by_block_id(self.blocks, block_id)
                # Remaining block budget
                block_budget = block.budget
                # Demand for this block
                demand_budget = task.budget_per_block[block_id]

                # todo: lock block
                allocation[i] = block_budget.allocate_budget(demand_budget)

        # sorted_tasks = []
        # for i in range(n_tasks):
        #     sorted_tasks.append(self.tasks[sorted_indices[i]])
        return allocation
