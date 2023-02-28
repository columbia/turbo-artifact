import time
from loguru import logger
from termcolor import colored
from typing import Dict, Optional

from precycle.executor import Executor
from precycle.task import Task, TaskInfo

from precycle.utils.utils import mlflow_log
from precycle.utils.utils import FINISHED, FAILED

from precycle.query_converter import QueryConverter
from precycle.budget.curves import ZeroCurve


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
        self.total_budget_spent_all_blocks = 0  # ZeroCurve()

    def try_run_task(self, task: Task) -> Optional[Dict]:
        """
        Try to run the task.
        If it can run, returns a metadata dict. Otherwise, returns None.
        """

        round = 0
        result = status = None
        run_metadata = {
            "sv_check_status": [],
            "sv_node_id": [],
            "run_types": [],
            "budget_per_block": [],
        }
        task.query = self.query_converter.convert_to_tensor(task.query)

        # Execute the plan to run the query # TODO: check if there is enough budget before running
        while not result and (not status or status == "sv_failed"):
            start_planning = time.time()
            # Get a DP execution plan for query.
            plan = self.planner.get_execution_plan(task)
            assert plan is not None
            planning_time = time.time() - start_planning
            # NOTE: if status is sth else like "out-of-budget" then it stops
            result, status = self.executor.execute_plan(plan, task, run_metadata)

            # Sanity checks
            # Second try must always use Laplaces so we can't reach third trial
            assert round < 2
            if round == 1:
                for run_type in run_metadata["run_types"][round].values():
                    assert run_type != "Histogram"
            round += 1

        if result:

            if self.config.logs.mlflow:
                # Log in MLFLOW to understand the bumps
                budget_per_block_list = run_metadata["budget_per_block"]
                for budget_per_block in budget_per_block_list:
                    for block, budget in budget_per_block.items():
                        self.total_budget_spent_all_blocks += budget.epsilon
                mlflow_log(f"AllBlocks", self.total_budget_spent_all_blocks, task.id)

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
