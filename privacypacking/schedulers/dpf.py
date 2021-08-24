from typing import List

from privacypacking.budget import ZeroCurve
from privacypacking.schedulers.scheduler import Scheduler
from privacypacking.utils.scheduling import dominant_shares


class DPFBlock:
    """
    A wrapper for the traditional-block.
    It holds a reference to the traditional-block but also has some additional DPF properties
    """

    def __init__(self, block, n):
        self.id = block.id
        self.budget = (
            ZeroCurve()
        )  # Will be gradually unlocking budget till we reach full capacity
        self.allocated_budget = ZeroCurve()  # Budget currently allocated by tasks
        self.block = block
        self.fair_share = self.block.initial_budget / n

    def unlock_budget(self):
        self.budget = self.budget.add_with_threshold(
            self.fair_share, self.block.initial_budget
        )
        # print("\n\nFair Share \n", self.fair_share)
        # print("\nUpdate budget\n", self.budget)
        # print("\nTotal budget capacity\n", self.block.initial_budget)
        # print("\n\n")


class DPF(Scheduler):
    # Static variable ; for each traditional block the scheduler creates and holds a corresponding dpf_block
    # that has additional information related to the DPF scheduler
    dpf_blocks = {}

    def __init__(self, tasks, blocks, config=None):
        super().__init__(tasks, blocks)
        self.config = config
        assert config is not None

    def update_dpf_blocks(self):
        for block_id, block in self.blocks.items():
            if block_id not in DPF.dpf_blocks:
                DPF.dpf_blocks[block_id] = DPFBlock(block, self.config.scheduler_N)

    def unlock_block_budgets(self):
        new_task = self.tasks[-1]
        for block_id in new_task.budget_per_block.keys():
            dpf_block = DPF.dpf_blocks[block_id]
            # Unlock budget for each alpha
            dpf_block.unlock_budget()

    def order(self) -> List[int]:
        """Sorts the tasks by dominant share"""

        n_tasks = len(self.tasks)

        def index_key(index):
            # Lexicographic order (the dominant share is the first component)
            return dominant_shares(self.tasks[index], self.blocks)

        # Task number i is high priority if it has small dominant share
        original_indices = list(range(n_tasks))
        sorted_indices = sorted(original_indices, key=index_key, reverse=True)
        return sorted_indices

    def can_run(self, task):
        """
        A task can run only if we can allocate the demand budget
        for all the blocks requested
        """
        for block_id, demand_budget in task.budget_per_block.items():
            dpf_block = DPF.dpf_blocks[block_id]
            available_budget = dpf_block.budget - dpf_block.allocated_budget
            if not available_budget.can_allocate(demand_budget):
                return False
        return True

    def consume_budgets(self, task):
        """
        Updates the budgets of each block requested by the task
        """
        for block_id, demand_budget in task.budget_per_block.items():
            dpf_block = DPF.dpf_blocks[block_id]
            dpf_block.allocated_budget += demand_budget
            # Consume traditional block's budget as well
            dpf_block.block.budget -= demand_budget

    def schedule(self) -> List[int]:
        allocated_task_ids = []

        # Update dpf_blocks in case new blocks arrived
        self.update_dpf_blocks()

        # Unlock budgets
        self.unlock_block_budgets()

        # Task indices sorted by smallest dominant share
        sorted_indices = self.order()

        # todo: lock blocks
        # Try and schedule tasks
        for i in sorted_indices:
            task = self.tasks[i]
            if self.can_run(task):
                self.consume_budgets(task)
                allocated_task_ids.append(task.id)

        return allocated_task_ids
