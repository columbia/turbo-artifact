from datetime import datetime

# import simpy.rt
import simpy

from privacypacking.simulator import Blocks, ResourceManager, Tasks
from privacypacking.utils.utils import *


class Simulator:
    def __init__(self, config):
        # self.env = simpy.rt.RealtimeEnvironment(factor=0.1, strict=False)
        self.env = simpy.Environment()
        self.config = config
        self.rm = ResourceManager(self.env, self.config)
        Blocks(self.env, self.rm)
        Tasks(self.env, self.rm)

    def run(self):
        start = datetime.now()

        # TODO: make this configurable
        self.env.run(until=200)
        # self.env.run()

        # Rough estimate of the scheduler's performance
        simulation_duration = (datetime.now() - start).total_seconds()

        logs = self.config.logger.get_log_dict(
            self.rm.scheduler.task_queue.tasks
            + list(self.rm.scheduler.tasks_info.allocated_tasks.values()),
            self.rm.scheduler.blocks,
            self.rm.scheduler.tasks_info,
            list(self.rm.scheduler.tasks_info.allocated_tasks.keys()),
            self.config,
            scheduling_time=simulation_duration,
        )

        # Saving locally too
        self.config.logger.log(
            self.rm.scheduler.task_queue.tasks
            + list(self.rm.scheduler.tasks_info.allocated_tasks.values()),
            self.rm.scheduler.blocks,
            self.rm.scheduler.tasks_info,
            list(self.rm.scheduler.tasks_info.allocated_tasks.keys()),
            self.config,
            scheduling_time=simulation_duration,
        )
        return global_metrics(logs)
