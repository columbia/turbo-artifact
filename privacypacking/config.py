import math
import os
import random
import uuid
from datetime import datetime
from functools import partial
from typing import List

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from numpy.lib.arraysetops import isin
from omegaconf import OmegaConf

from privacypacking.budget import Block, Task
from privacypacking.budget.block_selection import BlockSelectionPolicy
from privacypacking.budget.budget import Budget
from privacypacking.budget.task import UniformTask
from privacypacking.schedulers.utils import (
    DOMINANT_SHARES,
    TASK_BASED_BUDGET_UNLOCKING,
    THRESHOLD_UPDATING,
    TIME_BASED_BUDGET_UNLOCKING,
)
from privacypacking.utils.utils import *


# Configuration Reading Logic
class Config:
    def __init__(self, config):

        logger.info(f"Initializing config: {config}")

        # Just a basic configuration file that is not Turing-complete...
        default_omegaconf = OmegaConf.load(
            Path(__file__).parent.joinpath("conf/default.yaml")
        )
        custom_omegaconf = OmegaConf.create(config.get("omegaconf", {}))
        self.omegaconf = OmegaConf.merge(default_omegaconf, custom_omegaconf)
        logger.info(f"OmegaConf: {self.omegaconf}")

        # Rest of the configuration below
        self.config = config
        self.epsilon = self.omegaconf.epsilon
        self.delta = self.omegaconf.delta
        self.global_seed = self.omegaconf.global_seed
        random.seed(self.global_seed)
        np.random.seed(self.global_seed)

        # self.deterministic = config[DETERMINISTIC]
        # if self.deterministic:

        # BLOCKS
        # self.blocks_spec = config[BLOCKS_SPEC]
        self.initial_blocks_num = self.omegaconf.blocks.initial_num

        if self.omegaconf.scheduler.method == "offline":
            self.omegaconf.blocks.max_num = self.initial_blocks_num

        self.max_blocks = self.omegaconf.blocks.max_num
        self.block_arrival_interval = 1

        # self.block_arrival_frequency = self.blocks_spec[BLOCK_ARRIVAL_FRQUENCY]
        # if self.block_arrival_frequency[ENABLED]:
        #     self.block_arrival_frequency_enabled = True
        #     if self.block_arrival_frequency[POISSON][ENABLED]:
        #         self.block_arrival_poisson_enabled = True
        #         self.block_arrival_constant_enabled = False
        #         self.block_arrival_interval = self.block_arrival_frequency[POISSON][
        #             BLOCK_ARRIVAL_INTERVAL
        #         ]
        #     if self.block_arrival_frequency[CONSTANT][ENABLED]:
        #         self.block_arrival_constant_enabled = True
        #         self.block_arrival_poisson_enabled = False
        #         self.block_arrival_interval = self.block_arrival_frequency[CONSTANT][
        #             BLOCK_ARRIVAL_INTERVAL
        #         ]
        # else:
        #     self.block_arrival_frequency_enabled = False
        #     self.block_arrival_constant_enabled = False
        #     self.block_arrival_poisson_enabled = False

        # TASKS
        # TODO: clean up this part too, after we merge the Alibaba ingestion code
        # self.tasks_spec = config[TASKS_SPEC]
        # self.curve_distributions = self.tasks_spec[CURVE_DISTRIBUTIONS]
        # self.max_tasks = None

        # # Setting config for "custom" tasks
        # self.data_path = self.curve_distributions[CUSTOM][DATA_PATH]
        # self.data_task_frequencies_path = self.curve_distributions[CUSTOM][
        #     DATA_TASK_FREQUENCIES_PATH
        # ]
        # self.custom_tasks_init_num = self.curve_distributions[CUSTOM][INITIAL_NUM]
        # self.custom_tasks_sampling = self.curve_distributions[CUSTOM][SAMPLING]

        if self.omegaconf.tasks.sampling:
            self.data_path = REPO_ROOT.joinpath("data").joinpath(
                self.omegaconf.tasks.data_path
            )
            self.tasks_path = self.data_path.joinpath(self.omegaconf.tasks.tasks_path)
            self.task_frequencies_path = self.data_path.joinpath(
                "task_frequencies"
            ).joinpath(self.omegaconf.tasks.frequencies_path)

            with open(self.task_frequencies_path, "r") as f:
                self.task_frequencies_file = yaml.safe_load(f)
            assert len(self.task_frequencies_file) > 0

            self.task_arrival_interval = (
                self.block_arrival_interval
                / self.omegaconf.tasks.avg_num_tasks_per_block
            )

        else:
            # Read one CSV that contains all tasks
            self.data_path = REPO_ROOT.joinpath("data").joinpath(
                self.omegaconf.tasks.data_path
            )
            self.tasks = pd.read_csv(self.data_path)
            self.tasks_generator = self.tasks.iterrows()
            self.sum_task_interval = self.tasks["relative_submit_time"].sum()
            self.task_arrival_interval_generator = self.tasks[
                "relative_submit_time"
            ].iteritems()

        # self.task_arrival_frequency_enabled = False
        # self.task_arrival_frequency = self.tasks_spec[TASK_ARRIVAL_FREQUENCY]
        # if self.task_arrival_frequency[ENABLED]:
        #     self.task_arrival_frequency_enabled = True
        #     self.task_arrival_poisson_enabled = False
        #     self.task_arrival_constant_enabled = False
        #     self.task_arrival_actual_enabled = False

        #     if self.task_arrival_frequency[POISSON][ENABLED]:
        #         self.task_arrival_poisson_enabled = True
        #         self.task_arrival_interval = (
        #             self.block_arrival_interval
        #             / self.task_arrival_frequency[POISSON][AVG_NUMBER_TASKS_PER_BLOCK]
        #         )
        #     elif self.task_arrival_frequency[CONSTANT][ENABLED]:
        #         self.task_arrival_constant_enabled = True
        #         self.task_arrival_interval = (
        #             self.block_arrival_interval
        #             / self.task_arrival_frequency[CONSTANT][AVG_NUMBER_TASKS_PER_BLOCK]
        #         )
        #     elif self.task_arrival_frequency[ACTUAL][ENABLED]:
        #         self.task_arrival_actual_enabled = True

        # # SCHEDULER
        # self.scheduler = config[SCHEDULER_SPEC]
        # self.scheduler_method = self.scheduler[METHOD]
        # self.scheduler_metric = self.scheduler[METRIC]
        # self.scheduler_N = self.scheduler[N]
        # if DATA_LIFETIME in self.scheduler and self.block_arrival_constant_enabled:
        #     self.scheduler_data_lifetime = self.scheduler[DATA_LIFETIME]
        #     self.scheduler_budget_unlocking_time = self.scheduler_data_lifetime / (
        #         self.scheduler_N
        #     )
        # else:
        #     self.scheduler_budget_unlocking_time = self.scheduler[BUDGET_UNLOCKING_TIME]
        # self.scheduler_scheduling_wait_time = self.scheduler[SCHEDULING_WAIT_TIME]

        # if SOLVER in self.scheduler:
        #     self.scheduler_solver = self.scheduler[SOLVER]
        # else:
        #     self.scheduler_solver = None

        # # self.scheduler_threshold_update_mechanism = self.scheduler[
        # #     THRESHOLD_UPDATE_MECHANISM
        # # ]
        # self.new_task_driven_scheduling = False
        # self.time_based_scheduling = False
        # self.new_block_driven_scheduling = False
        # if self.scheduler_method == THRESHOLD_UPDATING:
        #     self.new_task_driven_scheduling = True
        #     self.new_block_driven_scheduling = True
        # elif self.scheduler_method == TIME_BASED_BUDGET_UNLOCKING:
        #     self.time_based_scheduling = True
        #     if self.scheduler_metric == DOMINANT_SHARES:
        #         logger.warning(
        #             f"Using DPF in batch scheduler mode with {self.scheduler_scheduling_wait_time}.\n This is not the original DPF algorithm unless T << expected_task_interarrival_time "
        #         )
        # else:
        #     self.new_task_driven_scheduling = True

        # LOGS
        # if LOG_FILE in config:
        #     self.log_file = f"{config[LOG_FILE]}.json"
        # else:
        #     self.log_file = f"{self.scheduler_method}_{self.scheduler_metric}/{datetime.now().strftime('%m%d-%H%M%S')}_{str(uuid.uuid4())[:6]}.json"

        # if CUSTOM_LOG_PREFIX in config:
        #     self.log_path = LOGS_PATH.joinpath(config[CUSTOM_LOG_PREFIX]).joinpath(
        #         self.log_file
        #     )
        # else:
        #     self.log_path = LOGS_PATH.joinpath(self.log_file)

        # self.log_every_n_iterations = config[LOG_EVERY_N_ITERATIONS]

    def dump(self) -> dict:
        d = self.config
        d["omegaconf"] = OmegaConf.to_container(self.omegaconf)
        return d

    def create_task(self, task_id: int) -> Task:

        task = None
        # Read custom task specs from files
        if self.omegaconf.tasks.sampling:
            if not hasattr(self, "task_specs"):
                self.task_specs = [
                    self.load_task_spec_from_file(f"{self.tasks_path}/{task_file}")
                    for task_file in self.task_frequencies_file.keys()
                ]

                self.task_frequencies = [
                    task_frequency
                    for task_frequency in self.task_frequencies_file.values()
                ]

            task_spec_index = np.random.choice(
                len(self.task_specs),
                1,
                p=self.task_frequencies,
            )[0]

            task_spec = self.task_specs[task_spec_index]

            if self.omegaconf.tasks.block_selection_policy:
                block_selection_policy = BlockSelectionPolicy.from_str(
                    self.omegaconf.tasks.block_selection_policy
                )
            else:
                block_selection_policy = task_spec.block_selection_policy
            assert block_selection_policy is not None

            task = UniformTask(
                id=task_id,
                profit=task_spec.profit,
                block_selection_policy=block_selection_policy,
                n_blocks=task_spec.n_blocks,
                budget=task_spec.budget,
                name=task_spec.name,
            )
        # Not sampling but reading actual tasks sequentially from one file
        else:
            _, task_row = next(self.tasks_generator)
            orders = {}
            parsed_alphas = task_row["alphas"].strip("][").split(", ")
            parsed_epsilons = task_row["rdp_epsilons"].strip("][").split(", ")

            for i, alpha in enumerate(parsed_alphas):
                alpha = float(alpha)
                epsilon = float(parsed_epsilons[i])
                orders[alpha] = epsilon

            task = UniformTask(
                id=task_id,
                profit=float(task_row["profit"]),
                block_selection_policy=BlockSelectionPolicy.from_str(
                    task_row["block_selection_policy"]
                ),
                n_blocks=int(task_row["n_blocks"]),
                budget=Budget(orders),
                name=task_row["task_name"],
            )

        assert task is not None
        return task

    def create_block(self, block_id: int) -> Block:
        return Block.from_epsilon_delta(
            block_id, self.epsilon, self.delta, alpha_list=self.omegaconf.alphas
        )

    def set_task_arrival_time(self):
        if self.omegaconf.tasks.sampling == POISSON:
            task_arrival_interval = partial(
                random.expovariate, 1 / self.task_arrival_interval
            )()
        elif self.omegaconf.tasks.sampling == CONSTANT:
            task_arrival_interval = self.task_arrival_interval

        else:
            _, task_arrival_interval = next(self.task_arrival_interval_generator)
            normalized_task_interval = task_arrival_interval / self.sum_task_interval
            task_arrival_interval = (
                normalized_task_interval * self.block_arrival_interval * self.max_blocks
            )
        return task_arrival_interval

    def set_block_arrival_time(self):
        return self.block_arrival_interval

    def get_initial_tasks_num(self) -> int:
        return self.omegaconf.tasks.initial_num

    def get_initial_blocks_num(self) -> int:
        return self.initial_blocks_num

    def load_task_spec_from_file(
        self, path: Path = PRIVATEKUBE_DEMANDS_PATH
    ) -> TaskSpec:

        with open(path, "r") as f:
            demand_dict = yaml.safe_load(f)
            orders = {}
            for i, alpha in enumerate(demand_dict["alphas"]):
                orders[alpha] = demand_dict["rdp_epsilons"][i]
            block_selection_policy = None
            if "block_selection_policy" in demand_dict:
                block_selection_policy = BlockSelectionPolicy.from_str(
                    demand_dict["block_selection_policy"]
                )

            # Select num of blocks
            if isinstance(demand_dict["n_blocks"], int):
                n_blocks = demand_dict["n_blocks"]
            elif isinstance(demand_dict["n_blocks"], str):
                n_blocks_requests = demand_dict["n_blocks"].split(",")
                num_blocks = [
                    n_blocks_request.split(":")[0]
                    for n_blocks_request in n_blocks_requests
                ]
                frequencies = [
                    n_blocks_request.split(":")[1]
                    for n_blocks_request in n_blocks_requests
                ]
                n_blocks = np.random.choice(
                    num_blocks,
                    1,
                    p=frequencies,
                )[0]

            # Select profit
            if "profit" in demand_dict:
                if isinstance(demand_dict["profit"], (int, float)):
                    profit = demand_dict["profit"]
                elif isinstance(demand_dict["profit"], str):
                    # TODO: not sure the typing makes sense here
                    profit_requests = demand_dict["profit"].split(",")
                    profits = [
                        profit_request.split(":")[0]
                        for profit_request in profit_requests
                    ]
                    frequencies = [
                        profit_request.split(":")[1]
                        for profit_request in profit_requests
                    ]
                    profit = np.random.choice(
                        profits,
                        1,
                        p=frequencies,
                    )[0]
            else:
                profit = 1

            task_spec = TaskSpec(
                profit=float(profit),
                block_selection_policy=block_selection_policy,
                n_blocks=int(n_blocks),
                budget=Budget(orders),
                name=os.path.basename(path),
            )
        assert task_spec is not None
        return task_spec
