"""
Model a privacy-resource-manager that grants privacy-budget resources to incoming tasks
according to a given scheduling mechanism.

ResourceManager has several block-resources each one of them having a privacy budget-capacity.
Incoming tasks arrive each one of them having a privacy budget-demand for one or more blocks.
Resources are non-replenishable.

ResourceManager owns a scheduling mechanism for servicing tasks according to a given policy.
"""

import argparse
import random

import numpy as np
import simpy.rt

from datetime import datetime
from privacypacking import simulator
from privacypacking.utils.utils import *
from privacypacking.config import Config


class Simulator:
    def __init__(self, config):
        # TODO: use discrete events instead of real time
        self.env = simpy.rt.RealtimeEnvironment(factor=0.1, strict=False)
        self.config = Config(config)

        if self.config.deterministic:
            random.seed(self.config.global_seed)
            np.random.seed(self.config.global_seed)

        self.rm = simulator.ResourceManager(self.env, self.config)
        simulator.Tasks(self.env, self.rm)
        simulator.Blocks(self.env, self.rm)

    def run(self):
        start = datetime.now()
        self.env.run()
        # Rough estimate of the scheduler's performance
        simulation_duration = (datetime.now() - start).total_seconds()

        logs = self.config.logger.get_log_dict(
            list(self.rm.scheduler.tasks.values()) + list(self.rm.scheduler.allocated_tasks.values()),
            self.rm.scheduler.blocks,
            list(self.rm.scheduler.allocated_tasks.keys()),
            self.config,
            scheduling_time=simulation_duration,
        )

        # Saving locally too
        self.config.logger.log(
            list(self.rm.scheduler.tasks.values()) + list(self.rm.scheduler.allocated_tasks.values()),
            self.rm.scheduler.blocks,
            list(self.rm.scheduler.allocated_tasks.keys()),
            self.config,
            scheduling_time=simulation_duration,
        )
        metrics = global_metrics(logs)

        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_file")
    args = parser.parse_args()

    with open(DEFAULT_CONFIG_FILE, "r") as default_config:
        config = yaml.safe_load(default_config)
    with open(args.config_file, "r") as user_config:
        user_config = yaml.safe_load(user_config)

    # Update the config file with the user-config's preferences
    update_dict(user_config, config)
    # pp.pprint(self.config)
    Simulator(config).run()

if __name__ == "__main__":
    main()
