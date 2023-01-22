import time
from loguru import logger
from termcolor import colored
from typing import Dict, Optional
from collections import namedtuple

from precycle.task import Task
from precycle.executor import Executor

from precycle.utils.utils import mlflow_log
from precycle.utils.utils import ALLOCATED, FAILED


TaskInfo = namedtuple(
    "TaskInfo", ["task", "status", "planning_time", "run_metadata", "result"]
)


class QueryProcessor:
    def __init__(self, db, cache, planner, budget_accountant, config):
        self.config = config

        self.db = db
        self.cache = cache
        self.planner = planner
        self.budget_accountant = budget_accountant
        self.executor = Executor(self.cache, self.db, config)

        self.tasks_info = {}  # TODO: make this persistent

    def try_run_task(self, task: Task) -> Optional[Dict]:
        """
        Try to run the task.
        If it can run, returns a metadata dict. Otherwise, returns None.
        """

        # Get a DP execution plan for query.
        # The plan returned here if not None is eligible for execution
        start_planning = time.time()
        plan = self.planner.get_execution_plan(task)
        planning_time = time.time() - start_planning

        if plan:
            logger.info(
                colored(
                    f"Plan of cost {plan.cost} for query {task.query_id} on blocks {task.blocks}: {plan}.",
                    "green",
                )
            )

            # Execute the plan to run the query
            result, run_budget_per_block, run_metadata = self.executor.execute_plan(
                plan, task
            )

            # Consume budget from blocks if necessary
            for blocks, run_budget in run_budget_per_block.items():
                self.budget_accountant.consume_blocks_budget(blocks, run_budget)

        if run_metadata:
            status = ALLOCATED
            for key, value in run_metadata.items():
                mlflow_log(f"{key}", value, task.id)

        else:
            status = FAILED
            run_metadata = None
            result = None

            logger.info(
                colored(
                    f"Can't run query {task.query_id} on blocks {task.blocks}.", "red"
                )
            )

        self.tasks_info[task.id] = TaskInfo(
            task, status, planning_time, run_metadata, result
        )
        return run_metadata

    # self.db.run_task()
    # query = self.query_pool.get_query(task.query_id)
    # true_result = HyperBlock(
    #     {key: self.blocks[key] for key in task.blocks}
    # ).run(query)
    # error = abs(true_result - result)
    # run_metadata["error"] = error
    # ---------------------------------------------- #
