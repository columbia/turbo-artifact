from datetime import datetime

import simpy
from privacypacking.simulator import Blocks, ResourceManager, Tasks
from privacypacking.utils.utils import get_logs


class Simulator:
    def __init__(self, omegaconf):
        self.env = simpy.Environment()

        self.omegaconf = omegaconf
        # Start the block and tasks consumers
        self.rm = ResourceManager(self.env, omegaconf)
        self.env.process(self.rm.start())

        # Start the block and tasks producers
        Blocks(self.env, self.rm)
        Tasks(self.env, self.rm)

    def run(self):
        start = datetime.now()

        self.env.run()

        # Rough estimate of the scheduler's performance
        simulation_duration = (datetime.now() - start).total_seconds()
        logs = get_logs(
            self.rm.scheduler.task_queue.tasks
            + list(self.rm.scheduler.tasks_info.allocated_tasks.values()),
            self.rm.scheduler.blocks,
            self.rm.scheduler.tasks_info,
            # list(self.rm.scheduler.tasks_info.allocated_tasks.keys()),
            self.omegaconf,
            scheduling_time=simulation_duration,
            scheduling_queue_info=self.rm.scheduler.scheduling_queue_info
            if hasattr(self.rm.scheduler, "scheduling_queue_info")
            else None,
        )
        return logs
