import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple, Union

from loguru import logger
from omegaconf import DictConfig
from simpy import Event

from privacypacking.budget import Block, Task
from privacypacking.budget.block_selection import NotEnoughBlocks
from privacypacking.schedulers.utils import ALLOCATED, FAILED, PENDING


# TODO: efficient data structure here? (We have indices)
class TaskQueue:
    def __init__(self):
        self.tasks = []


class TasksInfo:
    def __init__(self):
        self.allocated_tasks = {}
        self.allocated_resources_events = {}
        self.tasks_status = {}
        self.scheduling_time = {}
        self.creation_time = {}
        self.scheduling_delay = {}
        self.allocation_index = {}

    def dump(self):
        tasks_info = {
            "status": self.tasks_status,
            "scheduling_delay": self.scheduling_delay,
            "creation_time": self.creation_time,
            "scheduling_time": self.scheduling_time,
            "allocation_index": self.allocation_index,
        }
        return tasks_info


class Scheduler:
    def __init__(
        self,
        metric=None,
        verbose_logs=False,
        simulator_config: Optional[DictConfig] = None,
    ):
        self.metric = metric
        self.task_queue = TaskQueue()
        self.blocks = {}
        self.tasks_info = TasksInfo()
        self.simulation_terminated = False
        self.allocation_counter = 0
        if verbose_logs:
            logger.warning("Verbose logs. Might be slow and noisy!")
            # Counts the number of scheduling passes for each scheduling step (fixpoint)
            self.iteration_counter = defaultdict(int)

            # Stores metrics every time we recompute the scheduling queue
            self.scheduling_queue_info = []

        self.simulator_config = simulator_config
        self.omegaconf = simulator_config.scheduler if simulator_config else None
        self.alphas = None
        self.start_time = datetime.now()

    def consume_budgets(self, task):
        """
        Updates the budgets of each block requested by the task
        """
        for block_id, demand_budget in task.budget_per_block.items():
            block = self.blocks[block_id]
            block.budget -= demand_budget

    def now(self) -> Optional[float]:
        return self.env.now if hasattr(self, "env") else 0

    def allocate_task(self, task: Task) -> None:
        """
        Updates the budgets of each block requested by the task and cleans up scheduler's state
        """
        # Consume_budgets
        self.consume_budgets(task)
        # Clean/update scheduler's state
        self.tasks_info.tasks_status[task.id] = ALLOCATED
        self.tasks_info.allocated_resources_events[task.id].succeed()
        del self.tasks_info.allocated_resources_events[task.id]
        self.tasks_info.scheduling_time[task.id] = self.now()
        self.tasks_info.scheduling_delay[task.id] = (
            self.tasks_info.scheduling_time[task.id]
            - self.tasks_info.creation_time[task.id]
        )

        # TODO: do we really need to keep the whole task object?
        self.tasks_info.allocated_tasks[task.id] = task

        self.task_queue.tasks.remove(task)  # Todo: this takes linear time -> optimize

    def schedule_queue(self) -> List[int]:
        """Takes some tasks from `self.tasks` and allocates them
        to some blocks from `self.blocks`.
        Modifies the budgets of the blocks inplace.
        Returns:
            List[int]: the ids of the tasks that were scheduled
        """
        allocated_task_ids = []
        # Run until scheduling cycle ends
        converged = False
        cycles = 0
        while not converged:
            cycles += 1
            # Timeout if the physical time is too long
            duration_seconds = (datetime.now() - self.start_time).total_seconds()
            if (
                self.omegaconf.scheduler_timeout_seconds
                and self.omegaconf.scheduler_timeout_seconds < duration_seconds
            ):
                # raise TimeoutError(
                #     f"The scheduler took {duration_seconds} to schedule {self.allocation_counter} tasks in {cycles} cycles."
                # )
                logger.error(
                    f"The scheduler took {duration_seconds} to schedule {self.allocation_counter} tasks in {cycles} cycles."
                )
                sys.exit(1)

            # Sort the remaining tasks and try to allocate the first one
            sorted_tasks = self.order(self.task_queue.tasks)
            converged = True

            # logger.info(f"Pending tasks: {[t.id for t in sorted_tasks]}")

            # logger.info(f"Sorted tasks: {[st.id for st in sorted_tasks]}")
            # time.sleep(1)

            n_allocated_tasks = 0
            for task in sorted_tasks:
                if self.can_run(task):
                    # print("Allocated:", task.name, " - with blocks", task.n_blocks)

                    self.allocate_task(task)
                    allocated_task_ids.append(task.id)
                    self.tasks_info.allocation_index[task.id] = self.allocation_counter
                    self.allocation_counter += 1
                    n_allocated_tasks += 1

                    if self.omegaconf.log_warning_every_n_allocated_tasks and (
                        self.allocation_counter
                        % self.omegaconf.log_warning_every_n_allocated_tasks
                        == 0
                    ):
                        logger.warning(
                            f"Number of allocated tasks: {self.allocation_counter} at time {self.now()}"
                        )
                    if (
                        self.metric.is_dynamic()
                        and n_allocated_tasks
                        % self.omegaconf.metric_recomputation_period
                        == 0
                    ):
                        # We go back to the beginning of the while loop
                        converged = False
                        break
                # else:
                #     logger.debug(
                #         f"Task {task.id} cannot run. Demand budget: {task.budget_per_block}"
                #     )
        return allocated_task_ids

    def add_task(self, task_message: Tuple[Task, Event]):
        (task, allocated_resources_event) = task_message
        try:
            self.task_set_block_ids(task)
            logger.debug(
                 f"Task: {task.id} added to the scheduler at {self.now()}. Name: {task.name}. Blocks: {list(task.budget_per_block.keys())}"
            )
        except NotEnoughBlocks as e:
            # logger.warning(
            #     f"{e}\n Skipping this task as it can't be allocated. Will not count in the total number of tasks?"
            # )
            self.tasks_info.tasks_status[task.id] = FAILED
            return

        # Update tasks_info
        self.tasks_info.tasks_status[task.id] = PENDING
        self.tasks_info.allocated_resources_events[task.id] = allocated_resources_event
        self.tasks_info.creation_time[task.id] = self.now()
        self.task_queue.tasks.append(task)

        # Express the demands as a sparse matrix (for relevance metrics)
        if hasattr(self.metric, "compute_relevance_matrix"):
            self.task_queue.tasks[-1].build_demand_matrix(
                max_block_id=self.simulator_config.blocks.max_num
            )

    def add_block(self, block: Block) -> None:
        if block.id in self.blocks:
            raise Exception("This block id is already present in the scheduler.")
        self.blocks.update({block.id: block})

        # Support blocks with custom support
        if not self.alphas:
            self.alphas = block.initial_budget.alphas

    def get_num_blocks(self) -> int:
        num_blocks = len(self.blocks)
        return num_blocks

    def order(self, tasks: List[Task]) -> List[Task]:
        """Sorts the tasks by metric"""

        # The overflow is the same for all the tasks in this sorting pass
        if hasattr(self.metric, "compute_overflow"):
            logger.info("Precomputing the overflow for the whole batch")
            overflow = self.metric.compute_overflow(self.blocks, tasks)
        elif hasattr(self.metric, "compute_relevance_matrix"):
            # TODO: generalize to other relevance heuristics
            logger.info("Precomputing the relevance matrix for the whole batch")
            # for t in tasks:
            #     # We assume that there are no missing blocks. Otherwise, compute the max block id.
            #     t.pad_demand_matrix(n_blocks=self.get_num_blocks(), alphas=self.alphas)
            relevance_matrix = self.metric.compute_relevance_matrix(self.blocks, tasks)

        def task_key(task):
            if hasattr(self.metric, "compute_overflow"):
                return self.metric.apply(task, self.blocks, tasks, overflow)
            elif hasattr(self.metric, "compute_relevance_matrix"):
                return self.metric.apply(task, self.blocks, tasks, relevance_matrix)
            else:
                return self.metric.apply(task, self.blocks, tasks)

        if hasattr(self, "scheduling_queue_info"):
            # Compute the metrics separately to log the result
            metrics = {task.id: task_key(task) for task in tasks}

            def manual_task_key(task):
                return metrics[task.id]

            def short_manual_task_key(task):
                # Don't log the whole list for DPF
                m = metrics[task.id]
                if isinstance(m, list):
                    return m[0]
                return m

            sorted_tasks = sorted(tasks, reverse=True, key=manual_task_key)

            # TODO: add an option to log only the top k tasks?
            ids_and_metrics = [
                (task.id, short_manual_task_key(task)) for task in sorted_tasks
            ]

            # We might have multiple scheduling passes a the same time step
            scheduling_time = self.now()
            self.scheduling_queue_info.append(
                {
                    "scheduling_time": scheduling_time,
                    "iteration_counter": self.iteration_counter[scheduling_time],
                    "ids_and_metrics": ids_and_metrics,
                }
            )

            self.iteration_counter[scheduling_time] += 1

            return sorted_tasks
        return sorted(tasks, reverse=True, key=task_key)

    def can_run(self, task: Task) -> bool:
        """
        A task can run only if we can allocate the demand budget
        for all the blocks requested
        """
        for block_id, demand_budget in task.budget_per_block.items():
            if not block_id in self.blocks:
                return False
            block = self.blocks[block_id]
            if not block.budget.can_allocate(demand_budget):
                return False
        return True

    def task_set_block_ids(self, task: Task) -> None:
        # Ask the stateful scheduler to set the block ids of the task according to the task's constraints
        # try:
        selected_block_ids = task.block_selection_policy.select_blocks(
            blocks=self.blocks, task_blocks_num=task.n_blocks
        )
        # except NotEnoughBlocks as e:
        #     logger.warning(e)
        #     logger.warning(
        #         "Setting block ids to -1, the task will never be allocated.\n Should we count it in the total number of tasks?"
        #     )
        #     selected_block_ids = [-1]
        assert selected_block_ids is not None
        task.set_budget_per_block(selected_block_ids)
