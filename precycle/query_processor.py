import time
from loguru import logger
from termcolor import colored
from typing import Dict, Optional
from collections import defaultdict

from precycle.utils.utils import mlflow_log
from precycle.task import Task
from precycle.executor import Executor


FAILED = "failed"
PENDING = "pending"
ALLOCATED = "allocated"


# Logging all the tasks ever processed
class TasksInfo:
    def __init__(self):
        self.allocated_tasks = {}
        self.tasks_status = {}
        self.planning_time = defaultdict(lambda: None)
        self.run_metadata = {}
        self.n_allocated_tasks = 0


class QueryProcessor:
    def __init__(self, db, cache, planner, budget_accountant, config):
        self.config = config

        self.db = db
        self.cache = cache
        self.planner = planner
        self.budget_accountant = budget_accountant
        self.executor = Executor(self.cache, self.db, config)

        self.tasks_info = TasksInfo()

    def try_run_task(self, task: Task) -> Optional[Dict]:
        """
        Try to run the task. If it can run, returns a metadata dict. Otherwise, returns None.
        """
        # Get a DP execution plan for query
        start_planning = time.time()
        # The plan returned here if not None is eligible for execution
        plan = self.planner.get_execution_plan(task)
        if not plan:
            logger.info(
                colored(
                    f"Can't run query {task.query_id} on blocks {task.blocks}.",
                    "red",
                )
            )
            return

        logger.info(
            colored(
                f"Plan of cost {plan.cost} for query {task.query_id} on blocks {task.blocks}: {plan}.",
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
            self.tasks_info.tasks_status[task.id] = ALLOCATED
            self.tasks_info.allocated_tasks[task.id] = task
            self.tasks_info.run_metadata[task.id] = run_metadata
            self.tasks_info.n_allocated_tasks += 1

            for key, value in run_metadata.items():
                mlflow_log(f"{key}", value, task.id)
        else:
            # Otherwise the task stays in the queue, maybe more budget will be unlocked next time!
            logger.debug(f"Task {task.id} cannot run.")
        return run_metadata
