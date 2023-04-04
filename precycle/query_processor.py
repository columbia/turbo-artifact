import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
from loguru import logger
from termcolor import colored

from precycle.executor import Executor
from precycle.task import Task, TaskInfo
from precycle.utils.logs import compute_hit_scores
from precycle.utils.utils import FAILED, FINISHED, mlflow_log

SLIDING_WINDOW_LENGTH = 1_000

class QueryProcessor:
    def __init__(self, db, cache, planner, budget_accountant, config):
        self.config = config
        self.db = db
        self.cache = cache
        self.planner = planner
        self.budget_accountant = budget_accountant
        self.executor = Executor(self.cache, self.db, self.budget_accountant, config)

        self.tasks_info = []
        self.total_budget_spent_all_blocks = 0  # ZeroCurve()
        
        
        self.score_counters = defaultdict(int)
        self.score_sliding_windows = defaultdict(list)
        self.counter = 0
        
        self.score_thresholds = {
            0.5: False,
            0.9: False,
            0.95: False,
            0.99: False,
        }

    def try_run_task(self, task: Task) -> Optional[Dict]:
        """
        Try to run the task.
        If it can run, returns a metadata dict. Otherwise, returns None.
        """

        round = 0
        result = None
        status = None
        run_metadata = {
            "sv_check_status": [],
            "sv_node_id": [],
            "run_types": [],
            "budget_per_block": [],
            "laplace_hits": [],
            "pmw_hits": [],
        }


        # Execute the plan to run the query # TODO: check if there is enough budget before running
        while result is None and (not status or status == "sv_failed"):
            start_planning = time.time()

            # If the SV failed, don't risk it and go with Laplace everywhere
            force_laplace = False if round == 0 else True
            if force_laplace:
                logger.debug("Forcing Laplace because SV failed")

            # Get a DP execution plan for query.
            plan = self.planner.get_execution_plan(task, force_laplace=force_laplace)
            assert plan is not None
            planning_time = time.time() - start_planning
            # print("Planning", planning_time)

            # NOTE: if status is sth else like "out-of-budget" then it stops
            # Hmm status seems unused?
            result, status = self.executor.execute_plan(plan, task, run_metadata)
            logger.debug(
                colored(
                    f"Task: {task.id}, Query: {task.query_id}, on blocks: {task.blocks}, Plan: {plan}.",
                    "green",
                )
            )
            round += 1

        if result is not None:
 
            if self.config.logs.mlflow:
                
                hit_scores = compute_hit_scores(**run_metadata)
                
                for score_name, score_value in hit_scores.items():
                    if not np.isnan(score_value):
                        # Not so meaningful for cumulative SV or Laplace score
                        self.score_counters[f"cumulative_{score_name}"] += score_value
                                
                    # Sliding averages
                    if len(self.score_sliding_windows[score_name]) >= SLIDING_WINDOW_LENGTH:
                        self.score_sliding_windows[score_name].pop(0) # Drop oldest score
                    self.score_sliding_windows[score_name].append(score_value)
                    
                
                budget_per_block_list = run_metadata["budget_per_block"]
                for budget_per_block in budget_per_block_list:
                    for _, budget in budget_per_block.items():
                        self.total_budget_spent_all_blocks += budget.epsilon

                if self.counter % 100 == 0:
                    mlflow_log(f"AllBlocks", self.total_budget_spent_all_blocks, task.id)
                    
                    for score_name in hit_scores.keys():
                        # Hopefully at least one score is not Nan over the window, otherwise the mean is NaN
                        non_nan_scores = np.array([score for score in self.score_sliding_windows[score_name] if not np.isnan(score)])
                        if len(non_nan_scores) > 0:
                            sliding_score = np.mean(non_nan_scores)
                            
                            for score_threshold in self.score_thresholds.keys():
                                if self.score_thresholds[score_threshold] == False and sliding_score >= score_threshold:
                                    # We passed a threshold for the first time
                                    mlflow_log(f"sliding_{score_name}_threshold_{score_threshold}", self.counter, task.id)
                                    self.score_thresholds[score_threshold] = True
                            
                            mlflow_log(f"sliding_{score_name}", sliding_score, task.id)
                        mlflow_log(f"cumulative_{score_name}", self.score_counters[f"cumulative_{score_name}"], task.id)

            status = FINISHED
            # logger.info(
            #     colored(
            #         f"Task: {task.id}, Query: {task.query_id}, Cost of plan: {plan.cost}, on blocks: {task.blocks}, Plan: {plan}. ",
            #         "green",
            #     )
            # )
        else:
            status = FAILED
            logger.info(
                colored(
                    f"Task: {task.id}, Query: {task.query_id}, on blocks: {task.blocks}, can't run query.",
                    "red",
                )
            )
        self.counter += 1
        self.tasks_info.append(
            TaskInfo(task, status, planning_time, run_metadata, result).dump()
        )
        return run_metadata
