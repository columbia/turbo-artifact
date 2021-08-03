import random

import numpy as np

from privacypacking.base_simulator import BaseSimulator
from privacypacking.budget.block import Block
from privacypacking.budget.task import (
    create_gaussian_task,
    create_laplace_task,
    create_subsamplegaussian_task,
)
from privacypacking.offline.schedulers.greedy_heuristics import OfflineDPF
from privacypacking.offline.schedulers.simplex import Simplex
from privacypacking.utils.utils import *


class OfflineSimulator(BaseSimulator):
    def __init__(self, config):
        super().__init__(config)
        self.validate(config)

    def validate(self, config):
        pass

    def choose_blocks(self):
        return range(
            self.config.blocks_num
        )  # todo: a task has demands from all blocks for now; change this

    def prepare_tasks(self):
        tasks = []
        task_num = 0
        ######## Laplace Tasks ########
        block_ids = self.choose_blocks()
        noises = np.linspace(
            self.config.laplace_noise_start,
            self.config.laplace_noise_stop,
            self.config.laplace_num,
        )
        tasks += [
            create_laplace_task(task_num + i, self.config.blocks_num, block_ids, l)
            for i, l in enumerate(noises)
        ]
        task_num += self.config.laplace_num

        ######## Gaussian Tasks ########
        block_ids = self.choose_blocks()
        sigmas = np.linspace(
            self.config.gaussian_sigma_start,
            self.config.gaussian_sigma_stop,
            self.config.gaussian_num,
        )
        tasks += [
            create_gaussian_task(task_num + i, self.config.blocks_num, block_ids, s)
            for i, s in enumerate(sigmas)
        ]
        task_num += self.config.gaussian_num

        ######## SubSampleGaussian Tasks ########
        block_ids = self.choose_blocks()
        sigmas = np.linspace(
            self.config.subsamplegaussian_sigma_start,
            self.config.subsamplegaussian_sigma_stop,
            self.config.subsamplegaussian_num,
        )
        tasks += [
            create_subsamplegaussian_task(
                task_num + i,
                self.config.blocks_num,
                block_ids,
                self.config.subsamplegaussian_dataset_size,
                self.config.subsamplegaussian_batch_size,
                self.config.subsamplegaussian_epochs,
                s,
            )
            for i, s in enumerate(sigmas)
        ]

        random.shuffle(tasks)

        return tasks

    def prepare_blocks(self):
        blocks = [
            Block.from_epsilon_delta(
                i, self.config.renyi_epsilon, self.config.renyi_delta
            )
            for i in range(self.config.blocks_num)
        ]
        return blocks

    def prepare_scheduler(self, tasks, blocks):
        if self.config.scheduler == SIMPLEX:
            return Simplex(tasks, blocks)
        elif self.config.scheduler == OFFLINE_DPF:
            return OfflineDPF(tasks, blocks)

    def run(self):
        blocks = self.prepare_blocks()
        tasks = self.prepare_tasks()
        scheduler = self.prepare_scheduler(tasks, blocks)
        allocation = scheduler.schedule()
        self.config.plotter.plot(tasks, blocks, allocation)
