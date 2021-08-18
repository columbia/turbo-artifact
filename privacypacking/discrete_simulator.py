"""Generates blocks, tasks and manages them with a stateful scheduler.
The single-threaded scheduler is responsible for managing the state, 
no need to lock/mutex on blocks and tasks.
"""
import argparse
from itertools import count

import simpy
from loguru import logger

from privacypacking.budget.block import Block
from privacypacking.config import Config
from privacypacking.logger import Logger
from privacypacking.privacy_packing_simulator import schedulers
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
        self.no_more_blocks = False
        self.no_more_tasks = False

        # The three processes
        self.main_process = self.env.process(self.main())
        self.block_gen_process = self.env.process(self.block_gen())
        self.task_gen_process = self.env.process(self.task_gen())

    def main(self):

        while True:
            # Wait until something new happens
            yield self.new_block_event | self.new_task_event

            logger.debug(
                f"[{self.env.now}] Adding new blocks {self.new_blocks} or tasks {self.new_tasks}"
            )

            # Update the state of the scheduler with the new blocks/tasks
            while self.new_blocks:
                self.scheduler.add_block(self.new_blocks.pop())
            while self.new_tasks:
                self.scheduler.add_task(self.new_tasks.pop())

            # Schedule (it modifies the blocks) and update the list of pending tasks
            allocated_task_ids = self.scheduler.schedule()
            self.scheduler.update_allocated_tasks(allocated_task_ids)

            logger.debug(
                f"[{self.env.now}] Allocated the following task ids: {allocated_task_ids}"
            )

            # TODO: improve the log period + perfs
            self.logger.log(
                self.scheduler.tasks + list(self.scheduler.allocated_tasks.values()),
                self.scheduler.blocks,
                list(self.scheduler.allocated_tasks.keys()),
                self.config,
            )

            if self.no_more_blocks and self.no_more_tasks:
                # End the simulation in advance
                return

    def block_gen(self):
        logger.debug("Starting block gen.")
        if self.config.block_arrival_frequency_enabled:
            while True:
                # Sleep until it's time to create a new block
                block_arrival_interval = self.config.set_block_arrival_time()
                yield self.env.timeout(block_arrival_interval)

                # Initialize a fresh block with a new id
                block = Block.from_epsilon_delta(
                    block_id=next(self.block_count),
                    epsilon=self.config.epsilon,
                    delta=self.config.delta,
                )

                # Add the block to the queue and ping the scheduler
                logger.debug(f"[{self.env.now}] New block: {block}")
                self.new_blocks.append(block)
                self.new_block_event.succeed()

                # Refresh the event to be ready for the next block
                self.new_block_event = self.env.event()
        else:
            # Nothing to generate, run the scheduler once and terminate
            self.no_more_blocks = True
            self.new_block_event.succeed()
            self.new_block_event = self.env.event()
            return

    def task_gen(self):
        if self.config.task_arrival_frequency_enabled:
            while True:
                # Sleep until it's time to create a new task
                task_arrival_interval = self.config.set_task_arrival_time()
                yield self.env.timeout(task_arrival_interval)

                # Pick a curve, a number of blocks, and create the task
                # TODO: pick a profit too
                curve_distribution = self.config.set_curve_distribution()
                task_blocks_num = self.config.set_task_num_blocks(
                    self.scheduler.blocks, curve_distribution
                )
                task = self.config.create_task(
                    blocks=self.scheduler.blocks,
                    task_id=next(self.task_count),
                    curve_distribution=curve_distribution,
                    task_blocks_num=task_blocks_num,
                )

                # Add the task to the queue and ping the scheduler
                logger.debug(f"[{self.env.now}] New task: {task}")
                self.new_tasks.append(task)
                self.new_task_event.succeed()

                # Refresh the event to be ready for the next task
                self.new_task_event = self.env.event()
        else:
            self.no_more_tasks = True
            self.new_task_event.succeed()
            self.new_task_event = self.env.event()
            return


def run(config: dict):
    env = simpy.Environment()
    sim = Simulator(env, Config(config))
    env.run(until=15)


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
