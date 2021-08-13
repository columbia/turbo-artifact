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

import simpy.rt
import numpy as np
from privacypacking.base_simulator import BaseSimulator
from privacypacking.budget.block import Block
from privacypacking.budget.task import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
    UniformTask,
)
from privacypacking.online.block_selecting_policies.latest_first import LatestFirst
from privacypacking.online.schedulers import dpf, fcfs
from privacypacking.utils.utils import *

schedulers = {FCFS: fcfs.FCFS, DPF: dpf.DPF}


class ResourceManager:
    """
    A resource-manager has several blocks each one of them having a privacy-budget.
    While privacy-budgets are not replenishable in the sense that they can't be returned after used
    by a task additional blocks with privacy-budgets may arrive.

    The resource-manager has a traffic generator process that causes tasks to arrive and be granted resources.

    As a task consumes privacy budget resources the level of those resources goes down.
    A task must be granted "all" the resources that it demands or "nothing".
    """

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.blocks = {}
        self.archived_allocated_tasks = []
        self.scheduler = schedulers[self.config.scheduler_name]
        if self.config.deterministic:
            random.seed(self.config.global_seed)
            np.random.seed(self.config.global_seed)

        # To store the incoming task demands
        self.task_demands_queue = simpy.Store(self.env)

        # A ResourceManager has two persistent processes.
        # One that models the arrival of new resources - creates blocks statically for now
        self.generate_blocks()
        # env.process(self.generate_blocks())
        # One that models the distribution of resources to tasks
        env.process(self.schedule())

    def generate_blocks(self):
        """
        Generate blocks.
        Various configuration parameters determine the distribution of block
        arrival times as well as the privacy budget of each block.
        """
        block_id = count()
        # Create the initial number of blocks
        for _ in range(self.config.blocks_num):
            block_id_ = next(block_id)
            self.blocks[block_id_] = Block.from_epsilon_delta(
                block_id_, self.config.epsilon, self.config.delta
            )
        # If more are set to keep coming
        if self.config.block_arrival_interval is not None:
            # Determine block arrival interval
            block_arrival_interval = self.set_block_arrival_time()
            while True:
                block_id_ = next(block_id)
                self.blocks[block_id_] = Block.from_epsilon_delta(
                    block_id_, self.config.epsilon, self.config.delta
                )
                yield self.env.timeout(block_arrival_interval)
                print("Generated blocks ", self.blocks)
            # todo: add locks

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
        # yield self.env.timeout(30)
        while True:
            # Pick the next task demand from the queue
            task, allocated_resources_event = yield self.task_demands_queue.get()
            waiting_tasks.append((task, allocated_resources_event))

            # Try and schedule one or more of the waiting tasks
            tasks = [t[0] for t in waiting_tasks]
            s = self.scheduler(tasks, self.blocks, self.config)
            allocated_ids = (
                s.schedule()
            )  # schedule is triggered every time a new task arrives

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

    def __init__(self, env, resource_manager):
        self.env = env
        self.resource_manager = resource_manager
        self.config = resource_manager.config

        env.process(self.generate_tasks())

    def generate_tasks(self):
        """
        Generate tasks.
        Various configuration parameters determine the distribution of task
        arrival times as well as the demands of each task.
        """

        task_id = count()
        # Determine task arrival interval
        task_arrival_interval = self.set_task_arrival_time()
        while True:
            self.env.process(self.task(next(task_id)))
            yield self.env.timeout(task_arrival_interval)

    def task(self, task_id):
        """
        Generated task behavior. Sets its own demand, notifies resource manager of its existence,
        waits till it gets scheduled and then is executed
        """

        print("Generated task: ", task_id)
        task_blocks_num = self.set_task_blocks_request()
        curve_distribution = self.set_curve_distribution()
        task = self.set_task(task_id, curve_distribution, task_blocks_num)

        allocated_resources_event = self.env.event()
        # Wait till the demand-request has been delivered to the resource-manager
        yield self.resource_manager.task_demands_queue.put(
            (task, allocated_resources_event)
        )
        print("Task", task_id, "inserted demand")
        # Wait till the demand-request has been granted by the resource-manager
        yield allocated_resources_event

        # print("Task ", task_id, "start running")
        # yield self.env.timeout()
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

    def set_task_blocks_request(self):
        task_blocks_num = None
        if self.config.blocks_request_random_enabled:
            task_blocks_num = random.randint(
                1, self.config.blocks_request_random_max_num
            )
        elif self.config.blocks_request_constant_enabled:
            task_blocks_num = self.config.blocks_request_constant_num
        print("\n\ntasks blocks num", task_blocks_num)
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
        return curve

    def set_task(self, task_id, curve_distribution, task_blocks_num):
        task = None
        selected_block_ids = self.set_selected_block_ids(task_blocks_num)

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

    def set_selected_block_ids(self, task_blocks_num):
        selected_block_ids = None
        policy = self.config.block_selecting_policy
        if policy == LATEST_FIRST:
            selected_block_ids = LatestFirst(
                blocks=self.resource_manager.blocks, task_blocks_num=task_blocks_num
            ).select_blocks()
        # elif other policy
        assert selected_block_ids is not None
        return selected_block_ids


# TODO: use discrete events instead of real time
class OnlineSimulator(BaseSimulator):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        env = simpy.rt.RealtimeEnvironment(factor=0.1, strict=False)
        resource_manager = ResourceManager(env, self.config)
        Tasks(env, resource_manager)
        env.run()
