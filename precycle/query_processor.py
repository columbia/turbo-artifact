import time
from loguru import logger
from termcolor import colored
from typing import Dict, Optional

from precycle.executor import Executor
from precycle.task import Task, TaskInfo

from precycle.utils.utils import mlflow_log
from precycle.utils.utils import FINISHED, FAILED

from precycle.query_converter import QueryConverter


class QueryProcessor:
    def __init__(self, db, cache, planner, budget_accountant, config):
        self.config = config

        self.db = db
        self.cache = cache
        self.planner = planner
        self.budget_accountant = budget_accountant
        self.executor = Executor(self.cache, self.db, self.budget_accountant, config)

        self.query_converter = QueryConverter(self.config.blocks_metadata)
        self.tasks_info = []

    def try_run_task(self, task: Task) -> Optional[Dict]:
        """
        Try to run the task.
        If it can run, returns a metadata dict. Otherwise, returns None.
        """

        task.query = self.query_converter.convert_to_tensor(task.query)

        # Get a DP execution plan for query.
        start_planning = time.time()
        plan = self.planner.get_execution_plan(task)
        planning_time = time.time() - start_planning

        assert plan is not None
        result = status = None

        # Execute the plan to run the query # TODO: check if there is enough budget before running
        while not result and (not status or status == "sv_failed"):
            # TODO: if status is sth else like "out-of-budget" then stop
            result, run_metadata, status = self.executor.execute_plan(plan, task)

        if result:
            status = FINISHED
            logger.info(
                colored(
                    f"Task: {task.id}, Query: {task.query_id}, Cost of plan: {plan.cost}, on blocks: {task.blocks}, Plan: {plan}. ",
                    "green",
                )
            )
        else:
            status = FAILED
            logger.info(
                colored(
                    f"Task: {task.id}, Query: {task.query_id}, on blocks: {task.blocks}, can't run query.",
                    "red",
                )
            )

        self.tasks_info.append(
            TaskInfo(task, status, planning_time, run_metadata, result).dump()
        )
        return run_metadata
