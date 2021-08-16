from typing import List


class Scheduler:
    # TODO: What is config? A `Config` object?
    def __init__(self, tasks, blocks, config=None):
        self.tasks = tasks
        self.blocks = blocks
        self.config = config

    def schedule(self) -> List[int]:
        """Takes some tasks from `self.tasks` and allocates them
        to some blocks from `self.blocks`.

        Modifies the budgets of the blocks inplace.

        Returns:
            List[int]: the ids of the tasks that were scheduled
        """
        pass

    def order(self) -> List[int]:
        pass

    def can_run(self, task):
        """
        A task can run only if we can allocate the demand budget
        for all the blocks requested
        """
        for block_id, demand_budget in task.budget_per_block.items():
            block = self.blocks[block_id]
            if not block.budget.can_allocate(demand_budget):
                return False
        return True

    def consume_budgets(self, task):
        """
        Updates the budgets of each block requested by the task
        """
        for block_id, demand_budget in task.budget_per_block.items():
            block = self.blocks[block_id]
            block.budget -= demand_budget
