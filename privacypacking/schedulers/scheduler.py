import json
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger
from omegaconf import DictConfig
from simpy import Event
from termcolor import colored

from privacypacking.budget import Block, HyperBlock, Task
from data.covid19.covid19_queries.queries import QueryPool
from privacypacking.budget.block_selection import NotEnoughBlocks
from privacypacking.schedulers.utils import ALLOCATED, FAILED, PENDING
from privacypacking.utils.utils import REPO_ROOT, mlflow_log

from privacypacking.planner.ilp import ILP
from privacypacking.planner.max_cuts_planner import MinCutsPlanner
from privacypacking.planner.min_cuts_planner import MaxCutsPlanner

from privacypacking.cache.deterministic_cache import DeterministicCache
from privacypacking.cache.probabilistic_cache import ProbabilisticCache


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
        self.planning_time = defaultdict(lambda: None)
        self.tasks_submit_time = {}
        self.run_metadata = {}

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
        self.allocation_counter = 0
        if verbose_logs:
            logger.warning("Verbose logs. Might be slow and noisy!")
            # Counts the number of scheduling passes for each scheduling step (fixpoint)
            self.iteration_counter = defaultdict(int)
            # Stores metrics every time we recompute the scheduling queue
            self.scheduling_queue_info = []

        self.simulator_config = simulator_config
        self.omegaconf = simulator_config.scheduler if simulator_config else None
        self.initial_blocks_num = self.simulator_config.blocks.initial_num
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

        assert self.blocks_path
        assert self.queries_path
        assert self.blocks_metadata

        self.query_pool = QueryPool(
            self.blocks_metadata["attributes_domain_sizes"], self.queries_path
        )

        self.alphas = None
        self.start_time = datetime.now()
        self.allocated_task_ids = []
        self.n_allocated_tasks = 0

        self.variance_reduction = self.omegaconf.variance_reduction

        if self.omegaconf["cache"] == "ProbabilisticCache":
            self.cache = ProbabilisticCache(cache_cfg=self.omegaconf.pmw_cfg)
        elif self.omegaconf["cache"] == "DeterministicCache":
            self.cache = DeterministicCache(self.variance_reduction)
        else:
            NotImplementedError

        planner_args = {
            "variance_reduction": self.variance_reduction,
            "enable_caching": self.omegaconf.enable_caching,
            "enable_dp": self.omegaconf.enable_dp,
        }
        self.planner = globals()[self.omegaconf["planner"]](
            self.cache, self.blocks, planner_args
        )

        self.experiment_prefix = ""  # f"{self.simulator_config.repetition}/{self.omegaconf['cache']}/{self.omegaconf['planner']}/"

    def consume_budgets(self, blocks, budget):
        """
        Updates the budgets of each block requested by the task
        """
        for block_id in blocks:
            block = self.blocks[block_id]
            block.budget -= budget

    def now(self) -> Optional[float]:
        return self.env.now if hasattr(self, "env") else 0

    def update_allocated_task(self, task: Task, run_metadata: Dict = {}) -> None:
        """
        Cleans up scheduler's state
        """
        # Update task logs
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
        self.tasks_info.run_metadata[task.id] = run_metadata

        # Update scheduler global state
        self.allocation_counter += 1
        self.n_allocated_tasks += 1
        self.task_queue.tasks.remove(task)

    def restart_scheduling_cycle(self):
        """For dynamic relevance metrics"""
        return (
            self.metric.is_dynamic()
            and (self.n_allocated_tasks % self.omegaconf.metric_recomputation_period)
            == 0
        )

    def try_run_task(self, task: Task) -> Optional[Dict]:
        """
        Try to run the task. If it can run, returns a metadata dict. Otherwise, returns None.
        """

        # We are in the special case where tasks request intervals
        requested_blocks = sorted(task.blocks)
        block_tuple = (requested_blocks[0], requested_blocks[-1])
        assert len(requested_blocks) == block_tuple[1] - block_tuple[0] + 1

        start_planning = time.time()
        plan = self.planner.get_execution_plan(
            task.query_id, task.utility, task.utility_beta, requested_blocks
        )  # The plan returned here if not None is eligible for execution

        if not plan:
            return

        logger.info(
            colored(
                f"Got plan (cost={plan.cost}) for blocks {block_tuple}: {plan}",
                "yellow",
            )
        )
        
        # Execute the plan for running the query and consume budget if necessary
        result, run_metadata = self.execute_plan(plan)

        # ---------- Additional Metadata ---------- #
        run_metadata["planning_time"] = time.time() - start_planning
        run_metadata["result"] = result
        query = self.query_pool.get_query(task.query_id)
        true_result = HyperBlock(
            {key: self.blocks[key] for key in requested_blocks}
        ).run(query)
        error = abs(true_result - result)
        run_metadata["error"] = error
        # ---------------------------------------------- #

        return run_metadata

    def schedule_queue(self) -> List[int]:
        """Takes some tasks from `self.tasks` and allocates them
        to some blocks from `self.blocks`.
        Modifies the budgets of the blocks inplace.
        Returns:
            List[int]: the ids of the tasks that were scheduled
        """
        # Sort the remaining tasks and try to allocate them
        sorted_tasks = self.order(self.task_queue.tasks)

        for task in sorted_tasks:
            # Do not schedule tasks whose lifetime has been exceeded
            if (
                self.tasks_info.tasks_lifetime[task.id]
                < (self.get_num_blocks() - self.initial_blocks_num)
                - self.tasks_info.tasks_submit_time[task.id]
            ):
                # TODO: we should pop them of the queue then, no? Instead of sorting each time
                continue

            # Call the planner, cache, executes linear query. Returns None if the query can't run.
            run_metadata = self.try_run_task(task)
            if run_metadata:
                # Store logs and update the runqueue if the task ran
                self.update_allocated_task(task, run_metadata)
                for key, value in run_metadata.items():
                    mlflow_log(f"{self.experiment_prefix}{key}", value, task.id)
            else:
                # Otherwise the task stays in the queue, maybe more budget will be unlocked next time!
                logger.debug(f"Task {task.id} cannot run.")

            # Some schedulers need to repeat multiple scheduling cycles and re-sort the tasks each time
            if self.restart_scheduling_cycle():
                self.schedule_queue()

    def execute_plan(self, plan) -> Tuple[float, Dict]:
        """
        run_budget: the budget that will be consumed from the blocks after running the query
        """
        # TODO: pass the right alphas depending on the block sizes?

        query_id = plan.query_id
        query = self.query_pool.get_query(query_id)

        results = []
        for run_op in plan.l:
            block_ids = list(range(run_op.blocks[0], run_op.blocks[-1] + 1))
            hyperblock = HyperBlock({key: self.blocks[key] for key in block_ids})

            if self.omegaconf.enable_caching:  # Using cache
                result, run_budget, run_metadata = self.cache.run(
                    query_id=query_id,
                    query=query,
                    noise_std=run_op.noise_std,
                    hyperblock=hyperblock,
                )
            else:  # Not using cache
                if self.omegaconf.enable_dp:  # Add DP noise
                    result, run_budget, run_metadata = hyperblock.run_dp(
                        query, run_op.noise_std
                    )
                else:
                    result = hyperblock.run(query)
                    run_metadata = 0
                    run_budget = None

            if run_budget is not None:
                self.consume_budgets(block_ids, run_budget)

            run_metadata["hyperblock_size"] = hyperblock.size

            results += [result]

        if results:
            return sum(results), run_metadata
        return None, run_metadata

    def add_task(self, task_message: Tuple[Task, Event]):
        (task, allocated_resources_event) = task_message
        try:
            task.sample_n_blocks_and_profit()
            self.task_set_block_ids(task)
            logger.debug(
                f"Task: {task.id} added to the scheduler at {self.now()}. Name: {task.name}. "
                f"Blocks: {task.blocks}"
            )
        except NotEnoughBlocks as e:
            logger.warning(
                f"{e}\n Skipping this task as it can't be allocated. Will not count in the total number of tasks?"
            )
            self.tasks_info.tasks_status[task.id] = FAILED
            return

        # Update tasks_info
        self.tasks_info.tasks_status[task.id] = PENDING
        self.tasks_info.tasks_lifetime[task.id] = self.omegaconf.task_lifetime
        self.tasks_info.tasks_submit_time[task.id] = self.get_num_blocks()
        self.tasks_info.allocated_resources_events[task.id] = allocated_resources_event
        self.tasks_info.creation_time[task.id] = self.now()
        self.task_queue.tasks.append(task)

        # # Express the demands as a sparse matrix (for relevance metrics)
        # if hasattr(self.metric, "compute_relevance_matrix"):
        #     self.task_queue.tasks[-1].build_demand_matrix(
        #         max_block_id=self.simulator_config.blocks.max_num
        #     )

    def add_block(self, block: Block) -> None:
        if block.id in self.blocks:
            raise Exception("This block id is already present in the scheduler.")

        block.data_path = f"{self.blocks_path}/block_{block.id}.csv"
        block.load_histogram(self.blocks_metadata["attributes_domain_sizes"])
        # block.date = self.blocks_metadata["blocks"][str(block.id)]["date"]
        block.size = self.blocks_metadata["blocks"][str(block.id)]["size"]
        block.domain_size = self.blocks_metadata["domain_size"]
        # block.load_raw_data()
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

    def task_set_block_ids(self, task: Task) -> None:
        # Ask the stateful scheduler to set the block ids of the task according to the task's policy
        task.blocks = list(
            task.block_selection_policy.select_blocks(
                blocks=self.blocks, task_blocks_num=task.n_blocks
            )
        )
        assert task.blocks is not None
