import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

from privacypacking.budget import Block, Task
from privacypacking.budget.basic_budget import BasicBudget
from privacypacking.budget.block_selection import BlockSelectionPolicy
from privacypacking.budget.budget import Budget
from privacypacking.budget.task import UniformTask
from privacypacking.utils.utils import DEFAULT_CONFIG_FILE, REPO_ROOT


# Configuration Reading Logic
class Config:
    def __init__(self, config):

        logger.info(f"Initializing config: {config}")

        # Just a basic configuration file that is not Turing-complete...
        default_omegaconf = OmegaConf.load(DEFAULT_CONFIG_FILE)
        custom_omegaconf = OmegaConf.create(config.get("omegaconf", {}))
        self.omegaconf = OmegaConf.merge(default_omegaconf, custom_omegaconf)
        # print(self.omegaconf)

        # Rest of the configuration below
        self.global_seed = self.omegaconf.global_seed
        random.seed(self.global_seed)
        np.random.seed(self.global_seed)

        # BLOCKS
        self.initial_blocks_num = self.omegaconf.blocks.initial_num
        if self.omegaconf.scheduler.method == "offline":
            self.omegaconf.blocks.max_num = self.initial_blocks_num
        self.max_blocks = self.omegaconf.blocks.max_num
        self.max_tasks = None
        self.block_arrival_interval = 1

        # TASKS
        # Read one CSV that contains all tasks
        self.tasks_path = REPO_ROOT.joinpath("data").joinpath(self.omegaconf.tasks.path)
        self.tasks = pd.read_csv(self.tasks_path)
        self.max_tasks = len(self.tasks)
        self.tasks_generator = self.tasks.iterrows()
        self.task_arrival_interval_generator = self.tasks[
            "relative_submit_time"
        ].iteritems()

    def dump(self) -> dict:
        return {"omegaconf": OmegaConf.to_container(self.omegaconf)}

    def create_task(self, task_id: int) -> Task:
        task = None
        # Reading workload
        _, task_row = next(self.tasks_generator)
        # orders = {}
        # parsed_alphas = task_row["alphas"].strip("][").split(", ")
        # parsed_epsilons = task_row["rdp_epsilons"].strip("][").split(", ")

        # for i, alpha in enumerate(parsed_alphas):
        #     alpha = float(alpha)
        #     epsilon = float(parsed_epsilons[i])
        #     orders[alpha] = epsilon

        task = UniformTask(
            id=task_id,
            query_id=int(task_row["query_id"]),
            query_type=task_row["query_type"],
            profit=float(task_row["profit"]),
            block_selection_policy=BlockSelectionPolicy.from_str(
                task_row["block_selection_policy"]
            ),
            n_blocks=int(task_row["n_blocks"]),
            budget=BasicBudget(float(task_row["epsilon"])),
            # budget=RenyiBudget(parsed_epsilons[0]),
            name=task_row["task_name"],
        )

        assert task is not None
        return task

    def create_block(self, block_id: int) -> Block:
        block = Block(
            block_id,
            BasicBudget(self.omegaconf.epsilon),
        )
        # block = Block.from_epsilon_delta(
        #                 block_id,
        #                 self.omegaconf.epsilon,
        #                 self.omegaconf.delta,
        #                 alpha_list=self.omegaconf.alphas,
        #             )
        return block

    def set_task_arrival_time(self):
        if self.omegaconf.tasks.sampling == "poisson":
            task_arrival_interval = partial(
                random.expovariate, 1 / self.task_arrival_interval
            )()
        elif self.omegaconf.tasks.sampling == "constant":
            task_arrival_interval = self.task_arrival_interval

        else:
            _, task_arrival_interval = next(self.task_arrival_interval_generator)
        return task_arrival_interval

    def set_block_arrival_time(self):
        return self.block_arrival_interval

    def get_initial_tasks_num(self) -> int:
        return self.omegaconf.tasks.initial_num

    def get_initial_blocks_num(self) -> int:
        return self.initial_blocks_num
