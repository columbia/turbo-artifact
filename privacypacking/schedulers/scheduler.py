from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple
from loguru import logger
from omegaconf import DictConfig
from simpy import Event
from termcolor import colored
import json
from data.covid19.covid19_queries.queries import QueryPool
from privacypacking.budget import Block, Task, HyperBlock
from privacypacking.budget.block_selection import NotEnoughBlocks
from privacypacking.cache.cache import A, R
from privacypacking.cache.deterministic_cache import DeterministicCache
from privacypacking.cache.per_block_pmw import PerBlockPMW
from privacypacking.schedulers.utils import ALLOCATED, FAILED, PENDING
from privacypacking.utils.utils import REPO_ROOT


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
        self.with_cache_plans_ran = 0
        self.without_cache_plans_ran = 0
        self.without_cache_plan_result = {}
        self.with_cache_plan_result = {}
        self.realized_budget = 0

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
            self.simulator_config.blocks.path
        )
        self.queries_path = REPO_ROOT.joinpath("data").joinpath(
            self.simulator_config.tasks.queries_path
        )
        self.blocks_metadata = None
        with open(
            REPO_ROOT.joinpath("data").joinpath(self.simulator_config.blocks.metadata)
        ) as f:
            self.blocks_metadata = json.load(f)

        self.query_pool = QueryPool(self.blocks_metadata["domain_size"], self.queries_path)

        self.alphas = None
        self.start_time = datetime.now()
        self.allocated_task_ids = []
        self.n_allocated_tasks = 0

        # self.cache = cache.DeterministicCache(self.omegaconf.max_aggregations_allowed)
        self.cache = PerBlockPMW()

    def consume_budgets(self, blocks, budget):
        """
        Updates the budgets of each block requested by the task
        """
        for block_id in blocks:
            block = self.blocks[block_id]
            block.budget -= budget
            self.tasks_info.realized_budget += budget.epsilon

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

                bs_list = sorted(list(task.budget_per_block.keys()))
                bs_tuple = (bs_list[0], bs_list[-1])

                without_cache_plan = R(
                    query_id=task.query_id, blocks=bs_tuple, budget=task.budget
                )
                # Find a plan to run the query using caching
                plan = None
                if self.omegaconf.enable_caching:
                    plan = self.cache.get_execution_plan(
                        task.query_id, bs_list, task.budget
                    )
                elif self.can_run(task.budget_per_block):
                    plan = without_cache_plan

                print("\n")
                print(
                    colored(
                        f"Without Caching plan of query {task.query_id} on blocks {bs_tuple}:     {without_cache_plan}",
                        "green",
                    )
                )
                print(
                    colored(
                        f"With Caching Plan of query {task.query_id} on blocks {bs_tuple}:        {plan}",
                        "blue",
                    )
                )

                # Execute Plan
                if plan is not None:
                    result = self.execute_plan(plan)
                    print(
                        colored(
                            f"With Cache Noisy Result of query {task.query_id} on blocks {bs_tuple}: {result}",
                            "blue",
                        )
                    )
                    self.update_allocated_task(task)

                    # Run the without_cache_plan just to store the result - for experiments
                    if str(without_cache_plan) != str(plan):
                        self.tasks_info.with_cache_plan_result[task.id] = result
                        blocks = {key: self.blocks[b] for b in bs_list} 
                        result = HyperBlock(blocks).run(
                            self.query_pool.get_query(plan.query_id), plan.budget
                        )
                        print(
                            colored(
                                f"Without Cache Noisy Result of query {task.query_id} on blocks {bs_tuple}: {result}",
                                "green",
                            )
                        )
                    self.tasks_info.without_cache_plan_result[task.id] = result

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
                    print(
                        colored(
                            f"Task {task.id} cannot run. Demand budget: {task.budget_per_block}\n",
                            "blue",
                        )
                    )

        return self.allocated_task_ids

    def execute_plan(self, plan):
        result = None

        if isinstance(plan, R):  # Run Query
            print(f"plan.blocks: {plan.blocks}")
            block_ids = list(range(plan.blocks[0], plan.blocks[-1] + 1))

            # We could save hyperblocks so that we don't recompute their histograms
            blocks = {key: self.blocks[b] for b in bs_list} 
            result, budget = self.cache.run(
                query_id=plan.query_id,
                query=self.query_pool.get_query(plan.query_id),
                block=HyperBlock(blocks),
            )

            if budget is not None:
                self.consume_budgets(block_ids, budget)

        elif isinstance(plan, A):  # Aggregate Partial Results
            agglist = [self.execute_plan(x) for x in plan.l]
            result = sum(agglist) / len(agglist)
            # TODO: weighted average instead
            # TODO: blocks need to have a size
            # for aggregation here we average (results are fractions)

        else:
            logger.error("Execution: no such operator")
            exit(1)

        return result

    def add_task(self, task_message: Tuple[Task, Event]):
        (task, allocated_resources_event) = task_message
        try:
            task.sample_n_blocks_and_profit()
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

        # TODO: load the data directly when the resource manager creates the block?
        # I think I prefer the resource manager being dumb and just "creating" the block and corresponding csv (csvs in reality already exist though)
        # the scheduler can load data in memory or load as histogram for better performance if configured
        assert self.blocks_path is not None
        assert self.blocks_metadata is not None

        # TODO: Later on, store blocks as sparse histogram directly?
        # block.load_raw_data()
        block.load_histogram(self.blocks_metadata["attribute_domain_sizes"])
        block.date = self.blocks_metadata["blocks"][block.id]["date"]
        block.size = self.blocks_metadata["blocks"][block.id]["size"]

        self.blocks.update({block.id: block})

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

        blocks = {key: self.blocks[b] for b in demand.keys()} 
        return HyperBlock(blocks).can_run()
        
        
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
        task.set_budget_per_block(
            selected_block_ids, demands_tiebreaker=self.omegaconf.demands_tiebreaker
        )
