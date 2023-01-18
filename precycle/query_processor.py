import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

from loguru import logger
from termcolor import colored

from budget import Task
from utils.utils import mlflow_log

from executor import Executor

from planner.ilp import ILP
from planner.max_cuts_planner import MaxCutsPlanner
from planner.min_cuts_planner import MinCutsPlanner

from cache.deterministic_cache import DeterministicCache
from cache.probabilistic_cache import ProbabilisticCache

FAILED = "failed"
PENDING = "pending"
ALLOCATED = "allocated"

# Logging all the tasks ever processed
class TasksInfo:
    def __init__(self):
        self.allocated_tasks = {}
        self.allocated_resources_events = {}
        self.tasks_status = {}
        self.creation_time = {}
        self.allocation_index = {}
        self.tasks_lifetime = {}
        self.planning_time = defaultdict(lambda: None)
        self.tasks_submit_time = {}
        self.run_metadata = {}

    def dump(self):
        tasks_info = {
            "status": self.tasks_status,
            "creation_time": self.creation_time,
        }
        return tasks_info


class QueryProcessor:
    def __init__(self, psql_conn, budget_accountant, config):
        self.config = config
        self.psql_conn = psql_conn

        self.budget_accountant = budget_accountant

        self.tasks_info = TasksInfo()
        self.cache = DeterministicCache(config.cache)
        self.executor = Executor(self.psql_conn)

        self.start_time = datetime.now()
        self.allocated_task_ids = []
        self.n_allocated_tasks = 0

        # Initialize the Planner
        planner_args = {
            "enable_caching": self.config.enable_caching,
            "enable_dp": self.config.enable_dp,
        }
        self.planner = globals()[self.config.planner.method](
            self.cache, self.budget_accountant, planner_args
        )

    # Call the planner, cache, executes linear query. Returns None if the query can't run.
    def try_run_task(self, task: Task) -> Optional[Dict]:
        """
        Try to run the task. If it can run, returns a metadata dict. Otherwise, returns None.
        """
        # We are in the special case where tasks request intervals
        block_tuple = (task.blocks[0], task.blocks[-1])
        assert len(task.blocks) == block_tuple[1] - block_tuple[0] + 1

        # Get a DP execution plan for query
        start_planning = time.time()
        # The plan returned here if not None is eligible for execution
        plan = self.planner.get_execution_plan(task)

        if not plan:
            logger.info(
                colored(
                    f"Can't run query {task.query_id} on blocks {block_tuple}.",
                    "red",
                )
            )
            return

        logger.info(
            colored(
                f"Plan of cost {plan.cost} for query {task.query_id} on blocks {block_tuple}: {plan}.",
                "green",
            )
        )

        # Execute the plan to run the query and consume budget if necessary
        result, run_budget_per_block, run_metadata = self.executor.execute_plan(
            plan, task
        )

        # Consume budget from blocks
        for blocks, run_budget in run_budget_per_block.items():
            self.budget_accountant.consume_blocks_budget(blocks, run_budget)



        # ---------- Additional Metadata and Logging ---------- #
        run_metadata["planning_time"] = time.time() - start_planning
        run_metadata["result"] = result
        # query = self.query_pool.get_query(task.query_id)
        # true_result = HyperBlock(
        #     {key: self.blocks[key] for key in task.blocks}
        # ).run(query)
        # error = abs(true_result - result)
        # run_metadata["error"] = error
        # ---------------------------------------------- #

        if run_metadata:
            # Store logs and update the runqueue if the task ran
            self.update_allocated_task(task, run_metadata)

            for key, value in run_metadata.items():
                mlflow_log(f"{key}", value, task.id)
        else:
            # Otherwise the task stays in the queue, maybe more budget will be unlocked next time!
            logger.debug(f"Task {task.id} cannot run.")
        return run_metadata

    def update_allocated_task(self, task: Task, run_metadata: Dict = {}) -> None:
        # Update task logs
        self.tasks_info.tasks_status[task.id] = ALLOCATED
        self.tasks_info.allocated_tasks[task.id] = task
        self.allocated_task_ids.append(task.id)
        self.tasks_info.allocation_index[task.id] = self.allocation_counter
        self.tasks_info.run_metadata[task.id] = run_metadata
        self.n_allocated_tasks += 1
