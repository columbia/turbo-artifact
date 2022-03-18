from datetime import datetime

import simpy

from privacypacking.simulator import Blocks, ResourceManager, Tasks
from privacypacking.utils.utils import get_logs, global_metrics


class Simulator:
    def __init__(self, config):
        # self.env = simpy.rt.RealtimeEnvironment(factor=0.1, strict=False)
        self.env = simpy.Environment()
        self.config = config

        # Start the block and tasks consumers
        self.rm = ResourceManager(self.env, self.config)
        self.env.process(self.rm.start())

        # Start the block and tasks producers
        Blocks(self.env, self.rm)
        Tasks(self.env, self.rm)

    def run(self):
        start = datetime.now()

        # TODO: make this configurable
        # self.env.run(until=200)
        self.env.run()

        # Rough estimate of the scheduler's performance
        simulation_duration = (datetime.now() - start).total_seconds()

        logs = get_logs(
            self.rm.scheduler.task_queue.tasks
            + list(self.rm.scheduler.tasks_info.allocated_tasks.values()),
            self.rm.scheduler.blocks,
            self.rm.scheduler.tasks_info,
            # list(self.rm.scheduler.tasks_info.allocated_tasks.keys()),
            self.config,
            scheduling_time=simulation_duration,
            scheduling_queue_info=self.rm.scheduler.scheduling_queue_info
            if hasattr(self.rm.scheduler, "scheduling_queue_info")
            else None,
        )
        verbose = self.config.omegaconf.logs.save  # Saves tasks and blocks logs too
        return global_metrics(logs, verbose)
