import random
from typing import Iterable, List, Tuple

import numpy as np
from loguru import logger

from privacypacking.base_simulator import BaseSimulator
from privacypacking.budget import Block, Budget, Task
from privacypacking.budget.curves import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
)
from privacypacking.budget.task import UniformTask
from privacypacking.offline.schedulers.greedy_heuristics import (
    FlatRelevance,
    OfflineDPF,
    OverflowRelevance,
)
from privacypacking.offline.schedulers.simplex import Simplex
from privacypacking.utils import load_blocks_and_budgets_from_dir
from privacypacking.utils.utils import (
    CURVE_DISTRIBUTIONS,
    GAUSSIAN,
    LAPLACE,
    SUBSAMPLEGAUSSIAN,
    TASKS_SPEC,
)

# from privacypacking.utils.utils import OFFLINE_DPF, SIMPLEX


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

    def prepare_tasks_random_offset(
        self, blocks_and_budgets: Iterable[Tuple[int, "Budget"]]
    ) -> Iterable[Task]:
        tasks = []
        blocks_num = self.config.blocks_num
        for n_blocks, budget in blocks_and_budgets:
            if n_blocks < blocks_num:
                # Up to block_num - n_blocks (included)
                start = random.randint(0, blocks_num - n_blocks)
                stop = start + n_blocks - 1
                task = UniformTask(
                    id=len(tasks),
                    profit=1,
                    block_ids=range(start, stop + 1),
                    budget=budget,
                )
                tasks.append(task)
        return tasks

    def get_demand_block_ids(self, task_num_blocks: int) -> List[int]:
        total_num_blocks = self.config.blocks_num
        start = random.randint(0, total_num_blocks - task_num_blocks)
        stop = start + task_num_blocks - 1
        return list(range(start, stop + 1))

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

        # TODO: simplify the configuration format?
        tasks += [
            UniformTask(
                id=task_num + i,
                profit=1,
                block_ids=self.get_demand_block_ids(
                    int(
                        self.config.config["offline"][TASKS_SPEC][CURVE_DISTRIBUTIONS][
                            LAPLACE
                        ]["num_blocks"]
                    )
                ),
                budget=LaplaceCurve(noises[i]),
            )
            for i, l in enumerate(noises)
        ]
        task_num += self.config.laplace_num

        ######## Gaussian Tasks ########
        sigmas = np.linspace(
            self.config.gaussian_sigma_start,
            self.config.gaussian_sigma_stop,
            self.config.gaussian_num,
        )
        tasks += [
            UniformTask(
                id=task_num + i,
                profit=2,
                block_ids=self.get_demand_block_ids(
                    int(
                        self.config.config["offline"][TASKS_SPEC][CURVE_DISTRIBUTIONS][
                            GAUSSIAN
                        ]["num_blocks"]
                    )
                ),
                budget=GaussianCurve(s),
            )
            for i, s in enumerate(sigmas)
        ]
        task_num += self.config.gaussian_num

        ######## SubSampleGaussian Tasks ########
        sigmas = np.linspace(
            self.config.subsamplegaussian_sigma_start,
            self.config.subsamplegaussian_sigma_stop,
            self.config.subsamplegaussian_num,
        )
        tasks += [
            UniformTask(
                id=task_num + i,
                profit=4,
                block_ids=self.get_demand_block_ids(
                    int(
                        self.config.config["offline"][TASKS_SPEC][CURVE_DISTRIBUTIONS][
                            SUBSAMPLEGAUSSIAN
                        ]["num_blocks"]
                    )
                ),
                budget=SubsampledGaussianCurve.from_training_parameters(
                    self.config.subsamplegaussian_dataset_size,
                    self.config.subsamplegaussian_batch_size,
                    self.config.subsamplegaussian_epochs,
                    s,
                ),
            )
            for i, s in enumerate(sigmas)
        ]

        random.shuffle(tasks)

        return tasks

    def prepare_blocks(self):
        blocks = {}
        for i in range(self.config.blocks_num):
            blocks[i] = Block.from_epsilon_delta(
                i, self.config.renyi_epsilon, self.config.renyi_delta
            )
        return blocks

    def prepare_scheduler(self, tasks, blocks):
        try:
            scheduler_class = globals()[self.config.scheduler_name]
        except KeyError:
            logger.error(f"Unknown scheduler: {self.config.scheduler_name}")
        return scheduler_class(tasks, blocks)

        # if self.config.scheduler_name == SIMPLEX:
        #     return Simplex(tasks, blocks)
        # elif self.config.scheduler_name == OFFLINE_DPF:
        #     # return OfflineDPF(tasks, blocks)
        #     return FlatRelevance(tasks, blocks)

    def run(self):
        blocks = self.prepare_blocks()

        # Load PrivateKube's macrobenchmark data
        # blocks_and_budgets = load_blocks_and_budgets_from_dir()[0:10]
        # tasks = self.prepare_tasks_random_offset(blocks_and_budgets)

        tasks = self.prepare_tasks()
        scheduler = self.prepare_scheduler(tasks, blocks)
        allocation = scheduler.schedule()
        self.config.logger.log(tasks, blocks, allocation)
