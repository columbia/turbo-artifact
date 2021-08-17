"""Generates blocks, tasks and manages them with a stateful scheduler.
The scheduler is responsible for managing the state, no need to lock/mutex here.
"""
import argparse
import random
import sys

# from simpy.events import AllOf, Event
import time
from functools import partial
from itertools import count

import numpy as np
import simpy
import simpy.rt
from loguru import logger

from privacypacking.block_selecting_policies import LatestFirst
from privacypacking.budget.block import Block
from privacypacking.budget.task import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
    UniformTask,
)
from privacypacking.config import Config
from privacypacking.logger import Logger
from privacypacking.privacy_packing_simulator import schedulers
from privacypacking.schedulers import dpf, fcfs, greedy_heuristics, simplex
from privacypacking.schedulers.scheduler import Scheduler
from privacypacking.utils.utils import *


class Simulator:
    def __init__(
        self,
        env: simpy.Environment,
        config: Config,
    ) -> None:
        # Global state
        self.env = env
        self.config = config
        self.new_blocks = []
        self.new_tasks = []
        self.logger = Logger(config.log_path, config.scheduler_name)

        # Initialize the scheduler
        initial_tasks, initial_blocks = self.config.create_initial_tasks_and_blocks()
        self.scheduler = schedulers[self.config.scheduler_name](
            initial_tasks,
            initial_blocks,
            self.config,
        )
        self.task_count = count(len(initial_tasks))
        self.block_count = count(len(initial_blocks))

        # Synchronization events
        self.new_block_event = self.env.event()
        self.new_task_event = self.env.event()

        # The three processes
        self.main_process = self.env.process(self.main())
        self.block_gen_process = self.env.process(self.block_gen())
        self.task_gen_process = self.env.process(self.task_gen())

    def main(self):

        while True:
            # Wait until something new happens
            yield self.new_block_event | self.new_task_event

            # Update the state of the scheduler with the new blocks/tasks
            while self.new_blocks:
                self.scheduler.add_block(self.new_blocks.pop())
            while self.new_tasks:
                self.scheduler.add_task(self.new_tasks.pop())

            # Schedule (it modifies the blocks) and update the list of pending tasks
            allocated_task_ids = self.scheduler.schedule()
            self.scheduler.update_allocated_tasks(allocated_task_ids)

            # TODO: log period + perfs
            self.logger.log(
                self.scheduler.tasks + list(self.scheduler.allocated_tasks.values()),
                self.scheduler.blocks,
                list(self.scheduler.allocated_tasks.keys()),
                self.config,
            )

    def block_gen(self):
        while True:
            yield self.env.timeout(3)
            print(f"Done sleeping block, time to interrupt! {self.env.now}")
            self.new_block_event.succeed()
            self.new_block_event = self.env.event()

    def task_gen(self):
        while True:
            yield self.env.timeout(3)
            # TODO: task_count and create_task
            print(f"Done sleeping task, time to interrupt! {self.env.now}")
            self.new_task_event.succeed()
            self.new_task_event = self.env.event()


def run(config: dict):
    env = simpy.Environment()
    sim = Simulator(env)
    env.run(until=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_file")
    args = parser.parse_args()

    with open(DEFAULT_CONFIG_FILE, "r") as default_config:
        config = yaml.safe_load(default_config)
    with open(args.config_file, "r") as user_config:
        user_config = yaml.safe_load(user_config)

    # Update the config file with the user-config's preferences
    update_dict(user_config, config)

    run(config)
