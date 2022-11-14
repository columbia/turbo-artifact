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
from privacypacking.budget.curves import GaussianCurve
from privacypacking.budget.renyi_budget import RenyiBudget
from privacypacking.budget.task import UniformTask
from privacypacking.utils.utils import DEFAULT_CONFIG_FILE, REPO_ROOT


# Configuration Reading Logic
# TODO: why are the task utils here again? It doesn't seem to make sense.
# Maybe we can just pass omegaconf.tasks to Tasks?
class Config:
    def __init__(self, config):

        logger.info(f"Initializing config: {config}")

        # Just a basic configuration file that is not Turing-complete...
        default_omegaconf = OmegaConf.load(DEFAULT_CONFIG_FILE)
        custom_omegaconf = OmegaConf.create(config.get("omegaconf", {}))
        self.omegaconf = OmegaConf.merge(default_omegaconf, custom_omegaconf)
        # print(self.omegaconf)

        # Rest of the configuration below
        if self.omegaconf.enable_random_seed:
            random.seed(None)
            np.random.seed(None)
        else:
            random.seed(self.omegaconf.global_seed)
            np.random.seed(self.omegaconf.global_seed)

        # BLOCKS
        self.initial_blocks_num = self.omegaconf.blocks.initial_num
        if self.omegaconf.scheduler.method == "offline":
            self.omegaconf.blocks.max_num = self.initial_blocks_num
        self.max_blocks = self.omegaconf.blocks.max_num
        self.block_arrival_interval = 1

        # TASKS
        # Read one CSV that contains all tasks
        self.tasks_path = REPO_ROOT.joinpath("data").joinpath(self.omegaconf.tasks.path)
        self.tasks = pd.read_csv(self.tasks_path)

        if self.omegaconf.tasks.sampling:
            logger.info("Poisson sampling.")
            # Uniform sampling with Poisson arrival from the CSV file
            def row_sampler(df):
                while True:  # Don't stop, `max_tasks` will take care of that
                    d = df.sample(1)
                    yield 0, d.squeeze()  # Same signature as iterrows()

            self.tasks_generator = row_sampler(self.tasks)
            self.max_tasks = self.omegaconf.tasks.max_num
            self.initial_tasks_num = self.omegaconf.tasks.initial_num

        else:
            logger.info("Reading tasks in order with hardcoded arrival times.")
            # Browse tasks in order with hardcoded arrival times
            self.tasks_generator = self.tasks.iterrows()
            self.max_tasks = len(self.tasks)
            self.initial_tasks_num = 0
            self.task_arrival_interval_generator = self.tasks[
                "relative_submit_time"
            ].iteritems()

    def dump(self) -> dict:
        return {"omegaconf": OmegaConf.to_container(self.omegaconf)}

    def create_task(self, task_id: int, convert_to_rdp=True, gaussian=True) -> Task:
        # TODO: New omegaconf option for this? Or just use RDP really everywhere?
        _, task_row = next(self.tasks_generator)

        if "rdp_espilons" in task_row:
            orders = {}
            parsed_alphas = task_row["alphas"].strip("][").split(", ")
            parsed_epsilons = task_row["rdp_epsilons"].strip("][").split(", ")

            for i, alpha in enumerate(parsed_alphas):
                alpha = float(alpha)
                epsilon = float(parsed_epsilons[i])
                orders[alpha] = epsilon

            budget = RenyiBudget(parsed_epsilons[0])
        elif convert_to_rdp:
            epsilon, delta = float(task_row["epsilon"]), float(task_row["delta"])
            if gaussian:
                sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
                budget = GaussianCurve(sigma=sigma)
            else:
                # It doesn't really make sense (not a real mechanism) but this is just for debugging
                budget = RenyiBudget.from_epsilon_delta(epsilon, delta)
        else:
            budget = BasicBudget(float(task_row["epsilon"]))

        task = UniformTask(
            id=task_id,
            query_id=int(task_row["query_id"]),
            query_type=task_row["query_type"],
            profit=float(task_row["profit"]),
            block_selection_policy=BlockSelectionPolicy.from_str(
                task_row["block_selection_policy"]
            ),
            n_blocks=int(task_row["n_blocks"]),
            budget=budget,
            name=task_row["task_name"],
        )
        return task

    def create_block(self, block_id: int) -> Block:
        # block = Block(
        #     block_id,
        #     BasicBudget(self.omegaconf.epsilon),
        # )
        block = Block.from_epsilon_delta(
            block_id,
            self.omegaconf.epsilon,
            self.omegaconf.delta,
            alpha_list=self.omegaconf.alphas,
        )
        return block

    def set_task_arrival_time(self):
        if self.omegaconf.tasks.sampling == "poisson":
            task_arrival_interval = random.expovariate(
                1 / self.omegaconf.tasks.avg_num_tasks_per_block
            )
        elif self.omegaconf.tasks.sampling == "constant":
            task_arrival_interval = self.task_arrival_interval

        else:
            _, task_arrival_interval = next(self.task_arrival_interval_generator)
        return task_arrival_interval

    def set_block_arrival_time(self):
        return self.block_arrival_interval

    def get_initial_tasks_num(self) -> int:
        return self.initial_tasks_num

    def get_initial_blocks_num(self) -> int:
        return self.initial_blocks_num
