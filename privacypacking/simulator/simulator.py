"""
Model a privacy-resource-manager that grants privacy-budget resources to incoming tasks
according to a given scheduling mechanism.

ResourceManager has several block-resources each one of them having a privacy budget-capacity.
Incoming tasks arrive each one of them having a privacy budget-demand for one or more blocks.
Resources are non-replenishable.

ResourceManager owns a scheduling mechanism for servicing tasks according to a given policy.
"""

from datetime import datetime

import simpy.rt

from privacypacking.simulator import Blocks, ResourceManager, Tasks
from privacypacking.simulator.tasks import TasksFromFile
from privacypacking.utils.utils import *


class Simulator:
    def __init__(self, config):
        # self.env = simpy.rt.RealtimeEnvironment(factor=0.1, strict=False)
        self.env = simpy.Environment()

        self.config = config
        self.rm = ResourceManager(self.env, self.config)
        Blocks(self.env, self.rm)
        Tasks(self.env, self.rm)
        # TasksFromFile(
        #     self.env,
        #     self.rm,
        #     blocks_and_budgets_path=REPO_ROOT.joinpath("data/multiblock_dpf_killer"),
        # )

    def run(self):
        start = datetime.now()
        self.env.run(until=15)
        # Rough estimate of the scheduler's performance
        simulation_duration = (datetime.now() - start).total_seconds()

        logs = self.config.logger.get_log_dict(
            self.rm.scheduler.tasks + list(self.rm.scheduler.allocated_tasks.values()),
            self.rm.scheduler.blocks,
            list(self.rm.scheduler.allocated_tasks.keys()),
            self.config,
            scheduling_time=simulation_duration,
        )

        # Saving locally too
        self.config.logger.log(
            self.rm.scheduler.tasks + list(self.rm.scheduler.allocated_tasks.values()),
            self.rm.scheduler.blocks,
            list(self.rm.scheduler.allocated_tasks.keys()),
            self.config,
            scheduling_time=simulation_duration,
        )
        metrics = global_metrics(logs)

        return metrics
