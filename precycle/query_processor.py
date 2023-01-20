import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

from loguru import logger
from termcolor import colored

from precycle.task import Task
from precycle.utils.utils import mlflow_log

from precycle.executor import Executor

from precycle.planner.ilp import ILP
from precycle.planner.max_cuts_planner import MaxCutsPlanner
from precycle.planner.min_cuts_planner import MinCutsPlanner

from precycle.cache.deterministic_cache import DeterministicCache
from precycle.cache.probabilistic_cache import ProbabilisticCache

import typer
import psycopg2
from omegaconf import OmegaConf
from precycle.sql_converter import SQLConverter
from precycle.utils.utils import DEFAULT_CONFIG_FILE
from precycle.budget_accounant import BudgetAccountant


FAILED = "failed"
PENDING = "pending"
ALLOCATED = "allocated"

test = typer.Typer()


# Logging all the tasks ever processed
class TasksInfo:
    def __init__(self):
        self.allocated_tasks = {}
        self.tasks_status = {}
        self.allocation_index = {}
        self.planning_time = defaultdict(lambda: None)
        self.tasks_submit_time = {}
        self.run_metadata = {}

    def dump(self):
        tasks_info = {
            "status": self.tasks_status,
        }
        return tasks_info


class QueryProcessor:
    def __init__(self, psql_conn, budget_accountant, sql_converter, config):
        self.config = config
        self.psql_conn = psql_conn

        self.budget_accountant = budget_accountant
        self.sql_converter = sql_converter
        self.tasks_info = TasksInfo()
        self.cache = DeterministicCache(config.cache)
        self.executor = Executor(self.cache, self.psql_conn, sql_converter)

        self.start_time = datetime.now()
        self.allocated_task_ids = []
        self.n_allocated_tasks = 0

        # Initialize the Planner
        planner_args = {
            "enable_caching": self.config.enable_caching,
            "enable_dp": self.config.enable_dp,
            "cache_type": self.config.cache.type,
        }
        self.planner = globals()[self.config.planner.method](
            self.cache, self.budget_accountant, planner_args
        )

    # Call the planner, cache, executes linear query. Returns None if the query can't run.
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
        self.tasks_info.run_metadata[task.id] = run_metadata
        self.n_allocated_tasks += 1


@test.command()
def test(
    omegaconf: str = "precycle/config/precycle.json",
):
    omegaconf = OmegaConf.load(omegaconf)
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconf = OmegaConf.create(omegaconf)
    config = OmegaConf.merge(default_config, omegaconf)

    query_vector = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 2],
        [0, 0, 0, 3],
        [0, 0, 0, 4],
        [0, 0, 0, 5],
        [0, 0, 0, 6],
        [0, 0, 0, 7],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 2],
        [0, 0, 1, 3],
        [0, 0, 1, 4],
        [0, 0, 1, 5],
        [0, 0, 1, 6],
        [0, 0, 1, 7],
        [0, 0, 2, 0],
        [0, 0, 2, 1],
        [0, 0, 2, 2],
        [0, 0, 2, 3],
        [0, 0, 2, 4],
        [0, 0, 2, 5],
        [0, 0, 2, 6],
        [0, 0, 2, 7],
        [0, 0, 3, 0],
        [0, 0, 3, 1],
        [0, 0, 3, 2],
        [0, 0, 3, 3],
        [0, 0, 3, 4],
        [0, 0, 3, 5],
        [0, 0, 3, 6],
        [0, 0, 3, 7],
    ]
    num_requested_blocks = 1
    budget_accountant = BudgetAccountant(config=config.budget_accountant)
    num_blocks = budget_accountant.get_blocks_count()

    # Latest Blocks first
    requested_blocks = (num_blocks-num_requested_blocks, num_blocks-1)
    print(requested_blocks)

    task = Task(
        id=0,
        query_id=0,
        query_type="linear",
        query=query_vector,
        blocks=requested_blocks,
        n_blocks=num_requested_blocks,
        utility=100,
        utility_beta=0.0001,
        name=0,
    )

    # Initialize the PSQL connection
    try:
        # Connect to the PostgreSQL database server
        psql_conn = psycopg2.connect(
            host=config.postgres.host,
            database=config.postgres.database,
            user=config.postgres.username,
            password=config.postgres.password,
        )
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        exit(1)

    sql_converter = SQLConverter(config.blocks_server.block_metadata_path)

    # Initialize Query Processor
    query_processor = QueryProcessor(
        psql_conn, budget_accountant, sql_converter, config
    )
    run_metadata = query_processor.try_run_task(task)
    print(run_metadata)


if __name__ == "__main__":
    test()
