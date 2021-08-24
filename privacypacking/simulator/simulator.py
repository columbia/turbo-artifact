"""
Model a privacy-resource-manager that grants privacy-budget resources to incoming tasks
according to a given scheduling mechanism.

ResourceManager has several block-resources each one of them having a privacy budget-capacity.
Incoming tasks arrive each one of them having a privacy budget-demand for one or more blocks.
Resources are non-replenishable.

ResourceManager owns a scheduling mechanism for servicing tasks according to a given policy.
"""

import simpy.rt

from datetime import datetime
from privacypacking.simulator import Tasks, Blocks, ResourceManager
from privacypacking.utils.utils import *


class Simulator:
    def __init__(self, config):
        # TODO: use discrete events instead of real time
        self.env = simpy.rt.RealtimeEnvironment(factor=0.1, strict=False)
        self.config = config
        self.rm = ResourceManager(self.env, self.config)
        Tasks(self.env, self.rm)
        Blocks(self.env, self.rm)

    def run(self):
        start = datetime.now()
        self.env.run()
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
