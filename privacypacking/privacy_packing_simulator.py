"""
Model a privacy-resource-manager that grants privacy-budget resources to incoming tasks
according to a given scheduling mechanism.

ResourceManager has several block-resources each one of them having a privacy budget-capacity.
Incoming tasks arrive each one of them having a privacy budget-demand for one or more blocks.
Resources are non-replenishable.

ResourceManager owns a scheduling mechanism for servicing tasks according to a given policy.
"""

import random
from functools import partial
from itertools import count
from privacypacking.config import Config
import yaml
import simpy.rt
import numpy as np
import argparse
from privacypacking.budget.block import Block
from privacypacking.budget.task import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
    UniformTask,
)
from privacypacking.block_selecting_policies import LatestFirst
from privacypacking.schedulers import dpf, fcfs
from privacypacking.offline.schedulers import simplex
from privacypacking.utils.utils import *

schedulers = {FCFS: fcfs.FCFS, DPF: dpf.DPF, SIMPLEX: simplex.Simplex}


class ResourceManager:
    """
    A resource-manager has several blocks each one of them having a privacy-budget.
    While privacy-budgets are not replenishable in the sense that they can't be returned after used
    by a task additional blocks with privacy-budgets may arrive.

    The resource-manager has a traffic generator process that causes tasks to arrive and be granted resources.

    As a task consumes privacy budget resources the level of those resources goes down.
    A task must be granted "all" the resources that it demands or "nothing".
    """

    def __init__(self, environment, config):
        self.env = environment
        self.config = config
        self.blocks = {}
        self.archived_allocated_tasks = []
        self.total_init_tasks = (
            self.config.laplace_init_num
            + self.config.gaussian_init_num
            + self.config.subsamplegaussian_init_num
        )
        print(self.total_init_tasks)
        self.scheduler = schedulers[self.config.scheduler_name]
        if self.config.deterministic:
            random.seed(self.config.global_seed)
            np.random.seed(self.config.global_seed)

        # To store the incoming task demands
        self.task_demands_queue = simpy.Store(self.env)

        block_id_counter = count()
        # Create the initial number of blocks if such exists
        for _ in range(self.config.initial_blocks_num):
            block_id = next(block_id_counter)
            self.blocks[block_id] = Block.from_epsilon_delta(
                block_id, self.config.epsilon, self.config.delta
            )
        # A ResourceManager has two persistent processes.
        # One that models the arrival of new blocks
        self.env.process(self.generate_blocks())
        # One that models the distribution of resources to tasks
        self.env.process(self.schedule())

    def generate_blocks(self):
        """
        Generate blocks.
        Various configuration parameters determine the distribution of block
        arrival times as well as the privacy budget of each block.
        """

        block_id_counter = count(self.config.initial_blocks_num)
        # If more blocks are coming on the fly
        if self.config.block_arrival_frequency_enabled:
            # Determine block arrival interval
            block_arrival_interval = self.set_block_arrival_time()
            while True:
                block_id = next(block_id_counter)
                self.blocks[block_id] = Block.from_epsilon_delta(
                    block_id, self.config.epsilon, self.config.delta
                )
                yield self.env.timeout(block_arrival_interval)
                print("Generated blocks ", self.blocks)
        #     # todo: add locks

    def set_block_arrival_time(self):
        block_arrival_interval = None
        if self.config.block_arrival_poisson_enabled:
            block_arrival_interval = partial(
                random.expovariate, 1 / self.config.block_arrival_interval
            )
        elif self.config.block_arrival_constant_enabled:
            block_arrival_interval = self.config.block_arrival_interval
        assert block_arrival_interval is not None
        return block_arrival_interval

    def schedule(self):
        waiting_tasks = []
        while True:
            # Pick the next task demand from the queue
            task, allocated_resources_event = yield self.task_demands_queue.get()
            waiting_tasks.append((task, allocated_resources_event))

            # Don't schedule until all potential initial tasks have been collected
            if len(waiting_tasks) < self.total_init_tasks:
                continue

            # Try and schedule one or more of the waiting tasks
            tasks = [t[0] for t in waiting_tasks]
            s = self.scheduler(tasks, self.blocks, self.config)
            allocated_ids = s.schedule()

            # Update the logs for every time five new tasks arrive
            if task.id % 5 == 0:
                self.config.logger.log(
                    tasks + self.archived_allocated_tasks,
                    self.blocks,
                    allocated_ids
                    + [
                        allocated_task.id
                        for allocated_task in self.archived_allocated_tasks
                    ],
                    self.config,
                )
            print(
                "Scheduled tasks",
                [t[0].id for t in waiting_tasks if t[0].id in allocated_ids],
            )

            # Wake-up all the tasks that have been scheduled
            for task in waiting_tasks:
                if task[0].id in allocated_ids:
                    task[1].succeed()
                    self.archived_allocated_tasks += [task[0]]

            # todo: resolve race-condition between task-demands/budget updates and blocks; perhaps use mutex for quicker solution

            # Delete the tasks that have been scheduled from the waiting list
            waiting_tasks = [
                task for task in waiting_tasks if task[0].id not in allocated_ids
            ]


class Tasks:
    """
    Model task arrival rate and privacy demands.
    Each task's arrival time, privacy demands is determined by configuration.
    A new process is spawned for each task.
    """

    def __init__(self, environment, resource_manager):
        self.env = environment
        self.resource_manager = resource_manager
        self.config = resource_manager.config

        self.env.process(self.generate_tasks())

    def generate_tasks(self):
        """
        Generate tasks.
        Various configuration parameters determine the distribution of task
        arrival times as well as the demands of each task.
        """

        task_id_gen = count()
        # Create the initial number of tasks if such exists
        self.generate_initial_tasks(task_id_gen)

        # If more are set to keep coming
        if self.config.task_arrival_frequency_enabled:
            # Determine task arrival interval
            task_arrival_interval = self.set_task_arrival_time()
            while True:
                self.env.process(self.task(next(task_id_gen)))
                yield self.env.timeout(task_arrival_interval)

    def generate_initial_tasks(self, task_id_gen):
        tasks = (
            [LAPLACE] * self.config.laplace_init_num
            + [GAUSSIAN] * self.config.gaussian_init_num
            + [SUBSAMPLEGAUSSIAN] * self.config.subsamplegaussian_init_num
        )
        random.shuffle(tasks)
        for task in tasks:
            print(task)
            self.env.process(
                self.task(next(task_id_gen), task, self.set_task_num_blocks(task))
            )

    def task(self, task_id, curve_distribution=None, task_blocks_num=None):
        """
        Generated task behavior. Sets its own demand, notifies resource manager of its existence,
        waits till it gets scheduled and then is executed
        """

        print("Generated task: ", task_id)
        curve_distribution = (
            self.set_curve_distribution()
            if curve_distribution is None
            else curve_distribution
        )
        task_blocks_num = (
            self.set_task_num_blocks(curve_distribution)
            if task_blocks_num is None
            else task_blocks_num
        )

        task = self.create_task(task_id, curve_distribution, task_blocks_num)

        allocated_resources_event = self.env.event()
        # Wait till the demand-request has been delivered to the resource-manager
        yield self.resource_manager.task_demands_queue.put(
            (task, allocated_resources_event)
        )
        print("Task", task_id, "inserted demand")
        # Wait till the demand-request has been granted by the resource-manager
        yield allocated_resources_event
        print("Task ", task_id, "completed at ", self.env.now)

    def set_task_arrival_time(self):
        task_arrival_interval = None
        if self.config.task_arrival_poisson_enabled:
            task_arrival_interval = partial(
                random.expovariate, 1 / self.config.task_arrival_interval
            )
        elif self.config.task_arrival_constant_enabled:
            task_arrival_interval = self.config.task_arrival_interval
        assert task_arrival_interval is not None
        return task_arrival_interval

    def set_task_num_blocks(self, curve):
        task_blocks_num = None
        block_requests = self.config.curve_distributions[curve][BLOCKS_REQUEST]
        if block_requests[RANDOM][ENABLED]:
            task_blocks_num = random.randint(1, block_requests[RANDOM][NUM_BLOCKS_MAX])
        elif block_requests[CONSTANT][ENABLED]:
            task_blocks_num = block_requests[CONSTANT][NUM_BLOCKS]
        assert task_blocks_num is not None
        blocks_num = len(self.resource_manager.blocks)
        task_blocks_num = max(1, min(task_blocks_num, blocks_num))
        return task_blocks_num

    def set_curve_distribution(self):
        curve = np.random.choice(
            [GAUSSIAN, LAPLACE, SUBSAMPLEGAUSSIAN],
            1,
            p=[
                self.config.gaussian_frequency,
                self.config.laplace_frequency,
                self.config.subsamplegaussian_frequency,
            ],
        )
        return curve[0]

    def create_task(self, task_id, curve_distribution, task_blocks_num):
        task = None
        selected_block_ids = self.set_task_block_ids(
            task_blocks_num, curve_distribution
        )

        if curve_distribution == GAUSSIAN:
            sigma = random.uniform(
                self.config.gaussian_sigma_start, self.config.gaussian_sigma_stop
            )
            task = UniformTask(
                id=task_id,
                profit=1,
                block_ids=selected_block_ids,
                budget=GaussianCurve(sigma),
            )
        elif curve_distribution == LAPLACE:
            noise = random.uniform(
                self.config.laplace_noise_start, self.config.laplace_noise_stop
            )
            task = UniformTask(
                id=task_id,
                profit=1,
                block_ids=selected_block_ids,
                budget=LaplaceCurve(noise),
            )
        elif curve_distribution == SUBSAMPLEGAUSSIAN:
            sigma = random.uniform(
                self.config.subsamplegaussian_sigma_start,
                self.config.subsamplegaussian_sigma_stop,
            )
            task = UniformTask(
                id=task_id,
                profit=1,
                block_ids=selected_block_ids,
                budget=SubsampledGaussianCurve.from_training_parameters(
                    self.config.subsamplegaussian_dataset_size,
                    self.config.subsamplegaussian_batch_size,
                    self.config.subsamplegaussian_epochs,
                    sigma,
                ),
            )

        assert task is not None
        return task

    def set_task_block_ids(self, task_blocks_num, curve):
        selected_block_ids = None
        policy = self.config.curve_distributions[curve][BLOCK_SELECTING_POLICY]
        if policy == LATEST_FIRST:
            selected_block_ids = LatestFirst(
                blocks=self.resource_manager.blocks, task_blocks_num=task_blocks_num
            ).select_blocks()
        # elif other policy
        assert selected_block_ids is not None
        return selected_block_ids


DEFAULT_CONFIG_FILE = "privacypacking/config/default_config.yaml"

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
    # pp.pprint(self.config)

    # TODO: use discrete events instead of real time
    env = simpy.rt.RealtimeEnvironment(factor=0.1, strict=False)
    rm = ResourceManager(env, Config(config))
    Tasks(env, rm)
    env.run()
