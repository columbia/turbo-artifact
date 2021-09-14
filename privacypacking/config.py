import math
import os
import random
from datetime import datetime
from functools import partial
from typing import List

import numpy as np

from privacypacking.budget import Block, Task
from privacypacking.budget.curves import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
)
from privacypacking.budget.task import (
    UniformTask,
)
from privacypacking.logger import Logger
from privacypacking.budget.block_selection import LatestBlocksFirst, RandomBlocks
from privacypacking.utils.utils import *


# Configuration Reading Logic
class Config:
    def __init__(self, config):
        self.config = config
        self.epsilon = config[EPSILON]
        self.delta = config[DELTA]

        # DETERMINISM
        self.global_seed = config[GLOBAL_SEED]
        self.deterministic = config[DETERMINISTIC]
        if self.deterministic:
            random.seed(self.global_seed)
            np.random.seed(self.global_seed)

        # SCHEDULER
        self.scheduler = config[SCHEDULER_SPEC]
        self.scheduler_name = self.scheduler[NAME]
        self.scheduler_N = self.scheduler[N]

        # BLOCKS
        self.blocks_spec = config[BLOCKS_SPEC]
        self.initial_blocks_num = self.blocks_spec[INITIAL_NUM]
        self.block_arrival_frequency = self.blocks_spec[BLOCK_ARRIVAL_FRQUENCY]
        if self.block_arrival_frequency[ENABLED]:
            self.block_arrival_frequency_enabled = True
            if self.block_arrival_frequency[POISSON][ENABLED]:
                self.block_arrival_poisson_enabled = True
                self.block_arrival_constant_enabled = False
                self.block_arrival_interval = self.block_arrival_frequency[POISSON][
                    BLOCK_ARRIVAL_INTERVAL
                ]
            if self.block_arrival_frequency[CONSTANT][ENABLED]:
                self.block_arrival_constant_enabled = True
                self.block_arrival_poisson_enabled = False
                self.block_arrival_interval = self.block_arrival_frequency[CONSTANT][
                    BLOCK_ARRIVAL_INTERVAL
                ]
        else:
            self.block_arrival_frequency_enabled = False

        # TASKS
        self.tasks_spec = config[TASKS_SPEC]
        self.curve_distributions = self.tasks_spec[CURVE_DISTRIBUTIONS]

        # Setting config for "custom" tasks
        self.data_path = self.curve_distributions[CUSTOM][DATA_PATH]
        self.custom_tasks_init_num = self.curve_distributions[CUSTOM][INITIAL_NUM]
        self.custom_tasks_frequency = self.curve_distributions[CUSTOM][FREQUENCY]
        self.custom_tasks_sampling = self.curve_distributions[CUSTOM][SAMPLING]
        self.task_files = []
        if self.data_path != "":
            path = REPO_ROOT.joinpath("data").joinpath(self.data_path)
            for _, _, files in os.walk(path):
                self.task_files += [f'{path}/{file}' for file in files]
            random.shuffle(self.task_files)
            self.file_idx = -1
            assert len(self.task_files) > 0

        # Setting config for laplace tasks
        self.laplace = self.curve_distributions[LAPLACE]
        self.laplace_init_num = self.laplace[INITIAL_NUM]
        self.laplace_frequency = self.laplace[FREQUENCY]
        self.laplace_noise_start = self.laplace[NOISE_START]
        self.laplace_noise_stop = self.laplace[NOISE_STOP]

        # Setting config for gaussian tasks
        self.gaussian = self.curve_distributions[GAUSSIAN]
        self.gaussian_init_num = self.gaussian[INITIAL_NUM]
        self.gaussian_frequency = self.gaussian[FREQUENCY]
        self.gaussian_sigma_start = self.gaussian[SIGMA_START]
        self.gaussian_sigma_stop = self.gaussian[SIGMA_STOP]

        # Setting config for subsampledGaussian tasks
        self.subsamplegaussian = self.curve_distributions[SUBSAMPLEGAUSSIAN]
        self.subsamplegaussian_init_num = self.subsamplegaussian[INITIAL_NUM]
        self.subsamplegaussian_frequency = self.subsamplegaussian[FREQUENCY]
        self.subsamplegaussian_sigma_start = self.subsamplegaussian[SIGMA_START]
        self.subsamplegaussian_sigma_stop = self.subsamplegaussian[SIGMA_STOP]
        self.subsamplegaussian_dataset_size = self.subsamplegaussian[DATASET_SIZE]
        self.subsamplegaussian_batch_size = self.subsamplegaussian[BATCH_SIZE]
        self.subsamplegaussian_epochs = self.subsamplegaussian[EPOCHS]

        self.task_arrival_frequency = self.tasks_spec[TASK_ARRIVAL_FREQUENCY]
        if self.task_arrival_frequency[ENABLED]:
            self.task_arrival_frequency_enabled = True

            if self.task_arrival_frequency[POISSON][ENABLED]:
                self.task_arrival_poisson_enabled = True
                self.task_arrival_constant_enabled = False
                self.task_arrival_interval = self.task_arrival_frequency[POISSON][
                    TASK_ARRIVAL_INTERVAL
                ]

            elif self.task_arrival_frequency[CONSTANT][ENABLED]:
                self.task_arrival_constant_enabled = True
                self.task_arrival_poisson_enabled = False
                self.task_arrival_interval = self.task_arrival_frequency[CONSTANT][
                    TASK_ARRIVAL_INTERVAL
                ]
            assert (
                self.task_arrival_poisson_enabled != self.task_arrival_constant_enabled
            )
        else:
            self.task_arrival_frequency_enabled = False

        # Log file
        if LOG_FILE in config:
            self.log_file = f"{config[LOG_FILE]}.json"
        else:
            self.log_file = (
                f"{self.scheduler_name}/{datetime.now().strftime('%m%d-%H%M%S')}.json"
            )
        self.log_path = LOGS_PATH.joinpath(self.log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(self.log_path, self.scheduler_name)
        self.log_every_n_iterations = config[LOG_EVERY_N_ITERATIONS]

    def dump(self) -> dict:
        return self.config

    # Utils to initialize tasks and blocks. It only depends on the configuration, not on the simulator.
    def set_curve_distribution(self) -> str:
        curve = np.random.choice(
            [CUSTOM, GAUSSIAN, LAPLACE, SUBSAMPLEGAUSSIAN],
            1,
            p=[
                self.custom_tasks_frequency,
                self.gaussian_frequency,
                self.laplace_frequency,
                self.subsamplegaussian_frequency,
            ],
        )
        return curve[0]

    def set_task_num_blocks(self, curve: str, max_num_blocks: int = math.inf) -> int:
        task_blocks_num = None
        block_requests = self.curve_distributions[curve][BLOCKS_REQUEST]
        if block_requests[RANDOM][ENABLED]:
            task_blocks_num = random.randint(1, block_requests[RANDOM][NUM_BLOCKS_MAX])
        elif block_requests[CONSTANT][ENABLED]:
            task_blocks_num = block_requests[CONSTANT][NUM_BLOCKS]
        task_blocks_num = max(1, min(task_blocks_num, max_num_blocks))
        assert task_blocks_num is not None
        return task_blocks_num

    def get_block_selection_policy(self, curve: str):
        policy = self.curve_distributions[curve][BLOCK_SELECTING_POLICY]
        if policy == LATEST_BLOLCKS_FIRST:
            policy = LatestBlocksFirst
        elif policy == RANDOM_BLOCKS:
            policy = RandomBlocks
        assert policy is not None
        return policy

    # ################################## Used by 'privacypacking/discrete_simulator.py' ##################################
    # def create_initial_tasks_and_blocks_from_data(
    #         self,
    # ) -> Tuple[List[Task], Dict[int, Block]]:
    #
    #     # Prepare the initial blocks
    #     initial_blocks = {}
    #     block_id_counter = count()
    #     for _ in range(self.initial_blocks_num):
    #         block_id = next(block_id_counter)
    #         initial_blocks[block_id] = self.create_block(block_id)
    #
    #     # Sample the initial tasks according to the file's distribution
    #     initial_tasks = []
    #     task_id_counter = count()
    #     data_path = REPO_ROOT.joinpath("data").joinpath(
    #         self.config["tasks_spec"]["from_file"]["data_path"]
    #     )
    #     task_distribution = load_task_distribution(data_path)
    #
    #     # Since the seed is fixed, increasing `initial_num` adds more tasks but keeps the
    #     # first tasks identical
    #     for _ in range(self.config["tasks_spec"]["initial_num"]):
    #         task_parameters = random.choice(task_distribution)
    #         policy = task_parameters.policy
    #
    #         selected_block_ids = policy.select_blocks(
    #             initial_blocks, task_parameters.n_blocks
    #         )
    #
    #         task = UniformTask(
    #             id=next(task_id_counter),
    #             profit=task_parameters.profit,
    #             block_ids=selected_block_ids,
    #             budget=task_parameters.budget,
    #         )
    #
    #         initial_tasks.append(task)
    #
    #     return initial_tasks, initial_blocks
    #
    # def create_initial_tasks_and_blocks(
    #         self,
    # ) -> Tuple[List[Task], Dict[int, Block]]:
    #
    #     if "data_path" in self.config["tasks_spec"]:
    #         return self.create_initial_tasks_and_blocks_from_data()
    #
    #     # Create the initial tasks and blocks
    #     initial_blocks = {}
    #     block_id_counter = count()
    #     for _ in range(self.initial_blocks_num):
    #         block_id = next(block_id_counter)
    #         initial_blocks[block_id] = self.create_block(block_id)
    #
    #     initial_tasks = []
    #     task_id_counter = count()
    #     curves = (
    #             [LAPLACE] * self.laplace_init_num
    #             + [GAUSSIAN] * self.gaussian_init_num
    #             + [SUBSAMPLEGAUSSIAN] * self.subsamplegaussian_init_num
    #     )
    #     random.shuffle(curves)
    #     # TODO: We need a way to allow the same curve to request different nb of blocks
    #     # TODO: plot on Jupyter
    #     for curve_distribution in curves:
    #         task_blocks_num = self.set_task_num_blocks(
    #             curve_distribution, len(initial_blocks)
    #         )
    #         policy = self.get_policy(curve_distribution)
    #         selected_block_ids = policy.select_blocks(
    #             blocks=initial_blocks, task_blocks_num=task_blocks_num
    #         )
    #
    #         task = self.create_task(
    #             next(task_id_counter),
    #             curve_distribution,
    #             selected_block_ids=selected_block_ids,
    #         )
    #
    #         initial_tasks.append(task)
    #
    #     return initial_tasks, initial_blocks
    #
    # ####################################################################################################################

    def create_task(
            self, task_id: int, curve_distribution: str, num_blocks: int
    ) -> Task:

        # Todo: read profit from config
        task = None

        if curve_distribution is None:
            # If curve is not pre-specified (as in offline setting) then sample one
            curve_distribution = self.config.set_curve_distribution()

        if curve_distribution == CUSTOM:
            # Read custom task specs from a file
            if self.custom_tasks_sampling:
                self.file_idx = random.randint(0, len(self.task_files) - 1)
            else:
                self.file_idx += 1
            file = self.task_files[self.file_idx]
            task_spec = load_task_spec_from_file(file)
            task = UniformTask(
                id=task_id,
                profit=task_spec.profit,
                block_selection_policy=task_spec.block_selection_policy,
                n_blocks=task_spec.n_blocks,
                budget=task_spec.budget,
            )
        else:
            # Sample the specs of the task
            # TODO: this is a limiting assumption. It forces us to use the same number of blocks for all tasks with the same type.
            task_num_blocks = self.config.set_task_num_blocks(
                curve_distribution, num_blocks
            )
            block_selection_policy = self.config.get_block_selection_policy(
                curve_distribution
            )

            if curve_distribution == GAUSSIAN:
                sigma = random.uniform(
                    self.gaussian_sigma_start, self.gaussian_sigma_stop
                )
                task = UniformTask(
                    id=task_id,
                    profit=1,
                    block_selection_policy=block_selection_policy,
                    n_blocks=task_num_blocks,
                    budget=GaussianCurve(sigma),
                )
            elif curve_distribution == LAPLACE:
                noise = random.uniform(
                    self.laplace_noise_start, self.laplace_noise_stop
                )
                task = UniformTask(
                    id=task_id,
                    profit=1,
                    block_selection_policy=block_selection_policy,
                    n_blocks=task_num_blocks,
                    budget=LaplaceCurve(noise),
                )
            elif curve_distribution == SUBSAMPLEGAUSSIAN:
                sigma = random.uniform(
                    self.subsamplegaussian_sigma_start,
                    self.subsamplegaussian_sigma_stop,
                )
                task = UniformTask(
                    id=task_id,
                    profit=1,
                    block_selection_policy=block_selection_policy,
                    n_blocks=task_num_blocks,
                    budget=SubsampledGaussianCurve.from_training_parameters(
                        self.subsamplegaussian_dataset_size,
                        self.subsamplegaussian_batch_size,
                        self.subsamplegaussian_epochs,
                        sigma,
                    ),
                )
        assert task is not None
        return task

    def create_block(self, block_id: int) -> Block:
        return Block.from_epsilon_delta(block_id, self.epsilon, self.delta)

    def set_task_arrival_time(self):
        task_arrival_interval = None
        if self.task_arrival_poisson_enabled:
            task_arrival_interval = partial(
                random.expovariate, 1 / self.task_arrival_interval
            )
        elif self.task_arrival_constant_enabled:
            task_arrival_interval = self.task_arrival_interval
        assert task_arrival_interval is not None
        return task_arrival_interval

    def set_block_arrival_time(self):
        block_arrival_interval = None
        if self.block_arrival_poisson_enabled:
            block_arrival_interval = partial(
                random.expovariate, 1 / self.block_arrival_interval
            )
        elif self.block_arrival_constant_enabled:
            block_arrival_interval = self.block_arrival_interval
        assert block_arrival_interval is not None
        return block_arrival_interval

    def get_initial_task_curves(self) -> List[str]:
        curves = (
                [LAPLACE] * self.laplace_init_num
                + [GAUSSIAN] * self.gaussian_init_num
                + [SUBSAMPLEGAUSSIAN] * self.subsamplegaussian_init_num
                + [CUSTOM] * self.custom_tasks_init_num
        )
        random.shuffle(curves)
        return curves

    def get_initial_tasks_num(self) -> int:
        return (
                self.laplace_init_num
                + self.gaussian_init_num
                + self.subsamplegaussian_init_num
                + self.custom_tasks_init_num
        )

    def get_initial_blocks_num(self) -> int:
        return self.initial_blocks_num
