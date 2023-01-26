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
        self.executor = Executor(self.cache, self.db, config)

        self.query_converter = QueryConverter(self.config.blocks_metadata)
        self.tasks_info = []  # TODO: make this persistent

    def try_run_task(self, task: Task) -> Optional[Dict]:
        """
        Try to run the task.
        If it can run, returns a metadata dict. Otherwise, returns None.
        """

        task.query = self.query_converter.convert_to_tensor(task.query)

        # Get a DP execution plan for query.
        # The plan returned here if not None is eligible for execution
        start_planning = time.time()
        plan = self.planner.get_execution_plan(task)
        planning_time = time.time() - start_planning

        if plan:
            status = FINISHED

            logger.info(
                colored(
                    f"Task: {task.id}, Query: {task.query_id}, Cost of plan: {plan.cost}, on blocks: {task.blocks}, Plan: {plan}.",
                    "green",
                )
            )

            # Execute the plan to run the query
            result, run_budget_per_block, run_metadata = self.executor.execute_plan(
                plan, task
            )

            # Consume budget from blocks if necessary
            for blocks, run_budget in run_budget_per_block.items():
                # logger.info(run_budget)
                self.budget_accountant.consume_blocks_budget(blocks, run_budget)

            # for key, value in run_metadata.items():
            # mlflow_log(f"{key}", value, task.id)

        else:
            status = FAILED
            run_metadata = None
            result = None

            logger.info(
                colored(
                    f"Task: {task.id}, Query: {task.query_id}, on blocks: {task.blocks}, can't run query.",
                    "red",
                )
            )

        for block in range(task.blocks[0], task.blocks[1] + 1):
            budget = self.budget_accountant.get_block_budget(block)
            mlflow_log(f"{block}", max(budget.epsilons), task.id)

        self.tasks_info.append(
            TaskInfo(task, status, planning_time, run_metadata, result).dump()
        )
        return run_metadata
