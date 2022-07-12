import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple, Union
from loguru import logger
from omegaconf import DictConfig
from simpy import Event
from privacypacking.cache import cache
from privacypacking.cache.queries import *
from privacypacking.budget import Block, Task
from privacypacking.budget.block_selection import NotEnoughBlocks
from privacypacking.schedulers.utils import ALLOCATED, FAILED, PENDING
from privacypacking.utils.utils import REPO_ROOT
from termcolor import colored
from time import sleep


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
        self.tasks_lifetime = {}
        self.tasks_submit_time = {}
        self.tasks_substitutions_num = {}
        self.tasks_allocated_substitutions = {}
        self.cached_subs = 0
        self.subs = 0
        self.cached_original = 0
        self.original = 0
        self.original_result = {}
        self.substitute_result = {}

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
        self.blocks_path = REPO_ROOT.joinpath("data").joinpath(
            self.simulator_config.blocks.data_path
        )
        self.alphas = None
        self.start_time = datetime.now()
        self.allocated_task_ids = []
        self.n_allocated_tasks = 0
        print(self.omegaconf.max_substitutes_allowed)
        self.cache = cache.Cache(self.omegaconf.max_substitutes_allowed)

    def consume_budgets(self, task):
        """
        Updates the budgets of each block requested by the task
        """
        for block_id, demand_budget in task.budget_per_block.items():
            block = self.blocks[block_id]
            block.budget -= demand_budget

    def now(self) -> Optional[float]:
        return self.env.now if hasattr(self, "env") else 0

    def update_allocated_task(self, task: Task) -> None:
        """
        Cleans up scheduler's state
        """
        # Clean/update scheduler's state
        self.tasks_info.tasks_status[task.id] = ALLOCATED
        self.tasks_info.allocated_resources_events[task.id].succeed()
        del self.tasks_info.allocated_resources_events[task.id]
        self.tasks_info.scheduling_time[task.id] = self.now()
        self.tasks_info.scheduling_delay[task.id] = (
            self.tasks_info.scheduling_time[task.id]
            - self.tasks_info.creation_time[task.id]
        )
        self.tasks_info.allocated_tasks[task.id] = task
        self.allocated_task_ids.append(task.id)
        self.tasks_info.allocation_index[task.id] = self.allocation_counter
        self.allocation_counter += 1
        self.n_allocated_tasks += 1
        self.task_queue.tasks.remove(task)

    def schedule_queue(self) -> List[int]:
        """Takes some tasks from `self.tasks` and allocates them
        to some blocks from `self.blocks`.
        Modifies the budgets of the blocks inplace.
        Returns:
            List[int]: the ids of the tasks that were scheduled
        """
        # Run until scheduling cycle ends
        converged = False
        cycles = 0

        while not converged:
            cycles += 1

            # Sort the remaining tasks and try to allocate them
            sorted_tasks = self.order(self.task_queue.tasks)
            converged = True

            for task in sorted_tasks:
                # Do not schedule tasks whose lifetime has been exceeded
                if (
                    self.tasks_info.tasks_lifetime[task.id]
                    < self.get_num_blocks() - self.tasks_info.tasks_submit_time[task.id]
                ):
                    continue
                print("\n\n\ncached_subs", self.tasks_info.cached_subs, "subs", self.tasks_info.subs, "original", self.tasks_info.original, "cached_original", self.tasks_info.cached_original)

                # See if there is enough budget to run original request
                can_run = self.can_run(task.budget_per_block)
                # cached = False
                bs = original_bs = sorted(list(task.budget_per_block.keys()))
                blocks = (bs[0], bs[-1])

                # Check if a same task has been cached before
                if (cached := self.cache.find_result(task.query_id, blocks) is not None) \
                        or (cached := self.cache.find_substitute_result(task.query_id, tuple(bs)) is not None):
                    print(colored(f"Found cached result for {blocks}", "cyan"))
                    self.update_allocated_task(task)
                    self.tasks_info.cached_original += 1

                # Search for substitutes availability (including original request)
                elif self.omegaconf.allow_block_substitution and not can_run:
                    print(colored(f"Getting query {task.query_id} "
                                  f"Substitutes for demand {sorted(list(task.budget_per_block.keys()))}", "blue",))

                    # Loop until finding a substitute for the blocks on which the task can run
                    for substitute in self.cache.get_substitute_blocks(task.query_id, task.query_type, original_bs, task.k):
                        print("substitute", substitute)
                        for block in substitute:
                            print(f"             block {block} - available - {self.blocks[block].remaining_budget}")
                        print(colored(f"    substitute {substitute}", "magenta"))
                        if cached := self.cache.find_substitute_result(task.query_id, substitute) is not None:
                            print(colored(f"Found cached SUBSTITUTE result for {substitute}", "cyan"))
                            self.tasks_info.cached_subs += 1
                            task.budget_per_block = task.get_substitute_demand(substitute)
                            self.update_allocated_task(task)
                            break

                        else:
                            demand = task.get_substitute_demand(substitute)
                            bs = substitute
                            can_run = self.can_run(demand)
                            if can_run:
                                task.budget_per_block = demand
                                break
                # else:
                #     # See if there is enough budget to run original request
                #     can_run = self.can_run(task.budget_per_block)

                if not cached:
                    if can_run:
                        # print("Allocated:", task.name, " - with blocks", task.n_blocks)
                        self.consume_budgets(task)
                        self.update_allocated_task(task)

                        # Run task - update caches
                        result = self.run_task(task, bs)
                        if (
                            task.initial_budget_per_block == task.budget_per_block
                        ):  # if running on original blocks
                            print("Running on original blocks\n")
                            # Add result in cache and compute new distances
                            self.cache.add_result(task.query_id, blocks, task.budget.epsilon(0.0), result)
                            if self.omegaconf.allow_block_substitution:
                                self.cache.compute_distances(task.query_id, blocks, self.get_num_blocks(), task.k)
                            blocks = range(blocks[0], blocks[1] + 1)
                            self.tasks_info.original += 1
                            self.tasks_info.original_result[task.id] = result
                        else:
                            blocks = sorted(list(task.budget_per_block.keys()))
                            print(colored(f"        Found eligible Substitute {substitute}", "red",))
                            self.tasks_info.subs += 1
                            self.cache.add_substitute_result(task.query_id, substitute, task.budget.epsilon(0.0), result)
                            self.tasks_info.substitute_result[task.id] = result

                            # Run on original request just to store the result
                            temp = task.budget_per_block
                            task.budget_per_block = task.initial_budget_per_block
                            result = self.run_task(task, original_bs)
                            self.tasks_info.original_result[task.id] = result
                            task.budget_per_block = temp

                        for block in blocks:
                            if self.blocks[block].is_exhausted:
                                print(colored(f"        removing {block}", "yellow"))
                                self.cache.remove(block, self.get_num_blocks())

                        if (
                            self.metric.is_dynamic()
                            and self.n_allocated_tasks
                            % self.omegaconf.metric_recomputation_period
                            == 0
                        ):
                            # We go back to the beginning of the while loop
                            converged = False
                            break
                    else:
                        # logger.debug(
                        print(
                            colored(
                                f"Task {task.id} cannot run. Demand budget: {task.budget_per_block}\n",
                                "blue",
                            )
                        )
        return self.allocated_task_ids

    def run_task(self, task, blocks):
        df = []
        print(
            colored(f"Running query type {task.query_id} on blocks {blocks}", "green")
        )
        for block in blocks:
            df += [pd.read_csv(f"{self.blocks_path}/covid_block_{block}.csv")]
        df = pd.concat(df)
        res = globals()[f"query{task.query_id}"](df)
        print(
            colored(
                f"Result of query {task.query_id} on blocks {blocks}: \n{res}",
                "green",
            )
        )
        return res

    def add_task(self, task_message: Tuple[Task, Event]):
        (task, allocated_resources_event) = task_message
        try:
            self.task_set_block_ids(task)
            logger.debug(
                f"Task: {task.id} added to the scheduler at {self.now()}. Name: {task.name}. "
                f"Blocks: {list(task.budget_per_block.keys())}"
            )
        except NotEnoughBlocks as e:
            # logger.warning(
            #     f"{e}\n Skipping this task as it can't be allocated. Will not count in the total number of tasks?"
            # )
            self.tasks_info.tasks_status[task.id] = FAILED
            return

        # Update tasks_info
        self.tasks_info.tasks_status[task.id] = PENDING
        self.tasks_info.tasks_lifetime[task.id] = self.omegaconf.task_lifetime
        self.tasks_info.tasks_submit_time[task.id] = self.get_num_blocks()
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

    def can_run(self, demand) -> bool:
        """
        A task can run only if we can allocate the demand budget
        for all the blocks requested
        """
        for block_id, demand_budget in demand.items():
            if block_id not in self.blocks:
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
