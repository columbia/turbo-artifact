import random
import numpy as np
from privacypacking.base_simulator import BaseSimulator
from privacypacking.budget.block import create_block
from privacypacking.budget.task import create_laplace_task, create_gaussian_task, create_subsamplegaussian_task
from privacypacking.offline.schedulers.simplex import Simplex
from privacypacking.offline.schedulers.pierre_heuristic import PierreHeuristic
from privacypacking.utils.utils import *


class OfflineSimulator(BaseSimulator):

    def __init__(self, config):
        super().__init__(config)
        self.validate(config)

    def validate(self, config):
        pass

    def choose_blocks(self):
        return range(self.blocks_spec[NUM])  # todo: a task has demands from all blocks for now; change this

    def prepare_tasks(self):
        tasks = []
        task_num = 0
        tasks_spec = self.tasks_spec[CURVE_DISTRIBUTIONS]

        ######## Laplace Tasks ########
        laplace_tasks = tasks_spec[LAPLACE]
        block_ids = self.choose_blocks()
        noises = np.linspace(laplace_tasks[NOISE_START], laplace_tasks[NOISE_STOP], laplace_tasks[NUM])
        tasks += [create_laplace_task(task_num + i, self.blocks_spec[NUM], block_ids, l) for i, l in enumerate(noises)]
        task_num += laplace_tasks[NUM]

        ######## Gaussian Tasks ########
        gaussian_tasks = tasks_spec[GAUSSIAN]
        block_ids = self.choose_blocks()
        sigmas = np.linspace(gaussian_tasks[SIGMA_START], gaussian_tasks[SIGMA_STOP], gaussian_tasks[NUM])
        tasks += [create_gaussian_task(task_num + i, self.blocks_spec[NUM], block_ids, s) for i, s in enumerate(sigmas)]
        task_num += gaussian_tasks[NUM]

        ######## SubSampleGaussian Tasks ########
        subsamplegaussian_tasks = tasks_spec[SUBSAMPLEGAUSSIAN]
        block_ids = self.choose_blocks()
        sigmas = np.linspace(subsamplegaussian_tasks[SIGMA_START], subsamplegaussian_tasks[SIGMA_STOP],
                             subsamplegaussian_tasks[NUM])
        tasks += [create_subsamplegaussian_task(task_num + i, self.blocks_spec[NUM], block_ids,
                                                subsamplegaussian_tasks[DATASET_SIZE],
                                                subsamplegaussian_tasks[BATCH_SIZE],
                                                subsamplegaussian_tasks[EPOCHS], s)
                  for i, s in enumerate(sigmas)]

        random.shuffle(tasks)

        return tasks

    def prepare_blocks(self):
        blocks = [create_block(i, self.renyi_epsilon, self.renyi_delta) for i in range(self.blocks_spec[NUM])]
        return blocks

    def prepare_scheduler(self, tasks, blocks):
        if self.scheduler == SIMPLEX:
            return Simplex(tasks, blocks)
        elif self.scheduler == PIERRE_HEURISIC:
            return PierreHeuristic(tasks, blocks)

    def run(self):
        self.num_blocks = self.blocks_spec[NUM]
        blocks = self.prepare_blocks()
        tasks = self.prepare_tasks()
        scheduler = self.prepare_scheduler(tasks, blocks)
        allocation = scheduler.schedule()
        scheduler.plot(allocation)
