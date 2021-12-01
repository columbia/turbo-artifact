from typing import List, Tuple

from loguru import logger
from simpy import Event

from privacypacking.budget import Block, Budget, Task, ZeroCurve
from privacypacking.schedulers.scheduler import Scheduler

"""
For all schedulers based on gradually unlocking budget
"""


class UnlockingBlock(Block):
    def __init__(self, id: int, budget: Budget, n: int = 1):
        super().__init__(id, budget)
        self.unlocked_budget = (
            ZeroCurve()
        )  # Will be gradually unlocking budget till we reach full capacity
        self.fair_share = self.initial_budget / n

    def unlock_budget(self, budget: Budget = None):
        """Updates `self.unlocked_budget`. Fair share by default, but can use dynamic values too."""
        self.unlocked_budget = self.unlocked_budget.add_with_threshold(
            budget if budget else self.fair_share, self.initial_budget
        )
        # print("\n\nFair Share \n", self.fair_share)
        # print("\nUpdate budget\n", self.budget)
        # print("\nTotal budget capacity\n", self.block.initial_budget)
        # print("\n\n")

    def is_unlocked(self):
        return self.unlocked_budget == self.initial_budget

    @property
    def truncated_available_unlocked_budget(self) -> Budget:
        """Unlocked budget that is available for scheduling.
        truncated_available_unlocked_budget = positive [ unlocked_budget - ( initial_budget - remaining_budget) ]

        """
        return (self.unlocked_budget + self.budget - self.initial_budget).positive()

    @property
    def available_unlocked_budget(self) -> Budget:
        """Unlocked budget that is available for scheduling. Can be negative if the alpha is already consumed!
        available_unlocked_budget = unlocked_budget - ( initial_budget - remaining_budget)
        """
        return self.unlocked_budget + self.budget - self.initial_budget


class NBudgetUnlocking(Scheduler):
    """N-unlocking: unlocks some budget every time a new task arrives."""

    def __init__(self, metric, n):
        super().__init__(metric)
        self.n = n
        assert self.n is not None

    def add_task(self, task_message: Tuple[Task, Event]):
        super().add_task(task_message)
        self.unlock_block_budgets(self.task_queue.tasks)

    def add_block(self, block: Block) -> None:
        unlocking_block = UnlockingBlock(block.id, block.budget, self.n)
        super().add_block(unlocking_block)

    def unlock_block_budgets(self, tasks):
        new_task = tasks[-1]
        # Unlock budget only for blocks demanded by the last task
        for block_id in new_task.budget_per_block.keys():
            # Unlock budget for each alpha
            self.blocks[block_id].unlock_budget()

    def can_run(self, task):
        for block_id, demand_budget in task.budget_per_block.items():
            block = self.blocks[block_id]
            allocated_budget = block.initial_budget - block.budget
            available_budget = block.unlocked_budget - allocated_budget
            if not available_budget.can_allocate(demand_budget):
                return False
        return True


###########################################################################################
class TimeUnlockingBlock(UnlockingBlock):
    def __init__(self, id: int, budget: Budget, env, block_unlock_time, n: int = 1):
        super().__init__(id, budget, n)
        self.block_unlock_time = block_unlock_time
        self.env = env

    def wait_and_unlock(self):
        while not self.is_unlocked():
            yield self.env.timeout(self.block_unlock_time)
            logger.debug(
                "Unlock Block ",
                self.id,
                "locked",
                (self.initial_budget - self.unlocked_budget).epsilons,
            )
            self.unlock_budget()


class TBudgetUnlocking(Scheduler):
    """T-unlocking: unlocks some budget every time T units of time pass."""

    def __init__(
        self,
        metric,
        n,
        budget_unlocking_time,
        scheduling_wait_time,
        env,
        verbose_logs=False,
    ):
        super().__init__(metric, verbose_logs=verbose_logs)
        self.n = n
        self.budget_unlocking_time = budget_unlocking_time
        self.scheduling_wait_time = scheduling_wait_time
        self.env = env

        # TODO: why do we launch this as a process?
        self.env.process(self.schedule_queue())

    def add_block(self, block: Block) -> None:
        unlocking_block = TimeUnlockingBlock(
            block.id, block.budget, self.env, self.budget_unlocking_time, self.n
        )
        super().add_block(unlocking_block)
        self.env.process(unlocking_block.wait_and_unlock())

    def schedule_queue(self) -> List[int]:
        while not self.simulation_terminated:
            yield self.env.timeout(self.scheduling_wait_time)
            super().schedule_queue()

    def can_run(self, task):
        for block_id, demand_budget in task.budget_per_block.items():
            block = self.blocks[block_id]
            allocated_budget = block.initial_budget - block.budget
            available_budget = block.unlocked_budget - allocated_budget
            if not available_budget.can_allocate(demand_budget):
                return False
        return True
