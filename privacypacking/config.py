import random
from datetime import datetime
from functools import partial
from itertools import count
from typing import Dict, List, Tuple

import numpy as np

from privacypacking.block_selecting_policies import LatestFirst
from privacypacking.budget import Block, Task
from privacypacking.budget.task import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
    UniformTask,
)
from privacypacking.logger import Logger
from privacypacking.utils.utils import *


class Config:
    def __init__(self, config):
        self.config = config
        self.global_seed = config[GLOBAL_SEED]
        self.deterministic = config[DETERMINISTIC]
        self.epsilon = config[EPSILON]
        self.delta = config[DELTA]

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

        # Setting config for laplace tasks
        self.laplace = self.curve_distributions[LAPLACE]
        self.laplace_init_num = self.laplace[INITIAL_NUM]
        self.laplace_frequency = self.laplace[FREQUENCY]
        self.laplace_noise_start = self.laplace[NOISE_START]
        self.laplace_noise_stop = self.laplace[NOISE_STOP]

        self.gaussian = self.curve_distributions[GAUSSIAN]
        self.gaussian_init_num = self.gaussian[INITIAL_NUM]
        self.gaussian_frequency = self.gaussian[FREQUENCY]
        self.gaussian_sigma_start = self.gaussian[SIGMA_START]
        self.gaussian_sigma_stop = self.gaussian[SIGMA_STOP]

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
        log_path = LOGS_PATH.joinpath(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(log_path, self.scheduler_name)

    def dump(self) -> dict:
        return self.config

    # Utils to initialize tasks and blocks. It only depends on the configuration, not on the simulator.

    def set_curve_distribution(self):
        curve = np.random.choice(
            [GAUSSIAN, LAPLACE, SUBSAMPLEGAUSSIAN],
            1,
            p=[
                self.gaussian_frequency,
                self.laplace_frequency,
                self.subsamplegaussian_frequency,
            ],
        )
        return curve[0]

    def set_task_block_ids(self, blocks: Dict[int, Block], task_blocks_num, curve):
        selected_block_ids = None
        policy = self.curve_distributions[curve][BLOCK_SELECTING_POLICY]
        if policy == LATEST_FIRST:
            selected_block_ids = LatestFirst(
                blocks=blocks, task_blocks_num=task_blocks_num
            ).select_blocks()
        # elif other policy
        assert selected_block_ids is not None
        return selected_block_ids

    def set_task_num_blocks(self, blocks: Dict[int, Block], curve):
        task_blocks_num = None
        block_requests = self.curve_distributions[curve][BLOCKS_REQUEST]
        if block_requests[RANDOM][ENABLED]:
            task_blocks_num = random.randint(1, block_requests[RANDOM][NUM_BLOCKS_MAX])
        elif block_requests[CONSTANT][ENABLED]:
            task_blocks_num = block_requests[CONSTANT][NUM_BLOCKS]
        assert task_blocks_num is not None
        blocks_num = len(blocks)
        task_blocks_num = max(1, min(task_blocks_num, blocks_num))
        return task_blocks_num

    def create_initial_tasks_and_blocks(self) -> Tuple[List[Task], Dict[int, Block]]:
        # Create the initial tasks and blocks
        initial_blocks = {}
        block_id_counter = count()
        for _ in range(self.initial_blocks_num):
            block_id = next(block_id_counter)
            initial_blocks[block_id] = Block.from_epsilon_delta(
                block_id, self.epsilon, self.delta
            )

        initial_tasks = []
        task_id_counter = count()
        curves = (
            [LAPLACE] * self.laplace_init_num
            + [GAUSSIAN] * self.gaussian_init_num
            + [SUBSAMPLEGAUSSIAN] * self.subsamplegaussian_init_num
        )
        random.shuffle(curves)
        for curve_distribution in curves:
            task = self.create_task(
                next(task_id_counter),
                initial_blocks,
                curve_distribution,
                task_blocks_num=self.set_task_num_blocks(
                    initial_blocks, curve_distribution
                ),
            )

            initial_tasks.append(task)

        return initial_tasks, initial_blocks

    def create_task(
        self, blocks: Dict[int, Block], task_id, curve_distribution, task_blocks_num
    ):
        task = None
        selected_block_ids = self.set_task_block_ids(
            blocks,
            task_blocks_num,
            curve_distribution,
        )

        if curve_distribution == GAUSSIAN:
            sigma = random.uniform(self.gaussian_sigma_start, self.gaussian_sigma_stop)
            task = UniformTask(
                id=task_id,
                profit=1,
                block_ids=selected_block_ids,
                budget=GaussianCurve(sigma),
            )
        elif curve_distribution == LAPLACE:
            noise = random.uniform(self.laplace_noise_start, self.laplace_noise_stop)
            task = UniformTask(
                id=task_id,
                profit=1,
                block_ids=selected_block_ids,
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
                block_ids=selected_block_ids,
                budget=SubsampledGaussianCurve.from_training_parameters(
                    self.subsamplegaussian_dataset_size,
                    self.subsamplegaussian_batch_size,
                    self.subsamplegaussian_epochs,
                    sigma,
                ),
            )

        assert task is not None
        return task

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
