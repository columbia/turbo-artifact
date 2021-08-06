from typing import List


class Scheduler:
    def __init__(self, tasks, blocks, config=None):
        self.tasks = tasks
        self.blocks = blocks
        self.config = config

    def schedule(self):
        pass

    def order(self) -> List[int]:
        pass

    def can_run(self, task):
        """
        A task can run only if we can allocate the demand budget
        for all the blocks requested
        """
        for block_id, demand_budget in task.budget_per_block:
            block = self.blocks[block_id]
            if not block.budget.can_allocate(demand_budget):
                return False
        return True

    def consume_budgets(self, task):
        """
        Updates the budgets of each block requested by the task
        """
        for block_id, demand_budget in task.budget_per_block:
            block = self.blocks[block_id]
            block.budget -= demand_budget
