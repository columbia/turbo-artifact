"""
Model a privacy-resource-manager that grants privacy-budget resources to incoming tasks
according to a given scheduling mechanism.

ResourceManager has several block-resources each one of them having a privacy budget-capacity.
Incoming tasks arrive each one of them having a privacy budget-demand for one or more blocks.
Resources are non-replenishable.

ResourceManager owns a scheduling mechanism for servicing tasks according to a given policy.
"""

import random
from itertools import count

import simpy.rt

from privacypacking.base_simulator import BaseSimulator
from privacypacking.budget.block import Block
from privacypacking.budget.task import (
    create_gaussian_task,
    create_laplace_task,
    create_subsamplegaussian_task,
)
from privacypacking.online.schedulers import dpf, fcfs
from privacypacking.utils.utils import *

schedulers = {FCFS: fcfs.FCFS, DPF: dpf.DPF}


class ResourceManager:
    """
    A resource-manager has several blocks each one of them having a privacy-budget.
    While privacy-budgets are not replenishable in the sense that they can't be returned after used
    by a task more privacy budget may be added to them or additional blocks with privacy-budgets may arrive.

    The resource-manager has a traffic generator process that causes tasks to arrive and be granted resources.

    As a task consumes privacy budget resources the level of those resources goes down.
    A task must be granted "all" the resources that it demands or "nothing".
    """

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.blocks = []
        self.archived_allocated_tasks = []
        self.scheduler = schedulers[self.config.scheduler]

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
        # arrival_interval_dist = 100  #self.env.rand.expovariate, 1 / self.env.config['task.arrival_interval']

        # while True:
        # self.env.process(self.task(next(block_id)))
        self.blocks = [
            Block.from_epsilon_delta(
                next(block_id), self.config.renyi_epsilon, self.config.renyi_delta
            )
            for _ in range(self.config.blocks_num)
        ]
        print("Generated blocks ", self.blocks)
        # yield self.env.timeout(arrival_interval_dist)

    def schedule(self):
        waiting_tasks = []
        yield self.env.timeout(30)
        while True:
            # Pick the next task demand from the queue
            task, allocated_resources_event = yield self.task_demands_queue.get()
            waiting_tasks.append((task, allocated_resources_event))

            # Try and schedule one or more of the waiting tasks
            tasks = [t[0] for t in waiting_tasks]
            s = self.scheduler(tasks, self.blocks)
            allocation = s.schedule()
            # Update the figures
            self.config.plotter.plot(
                tasks + self.archived_allocated_tasks,
                self.blocks,
                allocation + [True] * len(self.archived_allocated_tasks),
            )
            print(
                "Scheduled tasks",
                [waiting_tasks[i][0].id for i, t in enumerate(allocation) if t],
            )

            # Wake-up all the tasks that have been scheduled
            for i, t in enumerate(allocation):
                if t:
                    waiting_tasks[i][1].succeed()
                    self.archived_allocated_tasks += [waiting_tasks[i][0]]

            print("Block budget", self.blocks[0].budget)
            # todo: resolve race-condition between task-demands/budget updates and blocks; perhaps use mutex for quicker solution

            ###################### Moved that inside the scheduler ######################
            # Update/consume block-budgets
            # todo: lock
            # for i, t in enumerate(allocation):
            #     if t:
            #         task = waiting_tasks[i][0]
            #         for block_id in task.block_ids:
            #             block_demand_budget = task.budget_per_block[block_id]
            #             # Get block with "block_id"
            #             block = get_block_by_block_id(self.blocks, block_id)
            #             block.budget -= block_demand_budget
            #############################################################################

            # Delete the tasks that have been scheduled from the waiting list
            waiting_tasks = [
                waiting_tasks[i] for i, t in enumerate(allocation) if not t
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
        self.config.task_arrival_interval = 10  # self.env.rand.expovariate, 1 / self.env.config['task.arrival_interval']

        while True:
            self.env.process(self.task(next(task_id)))
            yield self.env.timeout(self.config.task_arrival_interval)

    def task(self, task_id):
        """Generated task behavior."""

        print("Generated task: ", task_id)
        # Determine task
        curve_distribution = random.choice([GAUSSIAN, LAPLACE, SUBSAMPLEGAUSSIAN])

        # Determine demands
        task_blocks_num = 2
        blocks_num = self.resource_manager.config.blocks_num
        task_blocks_num = max(1, min(task_blocks_num, blocks_num))

        task = None
        if curve_distribution == GAUSSIAN:
            sigma = random.uniform(
                self.config.gaussian_sigma_start, self.config.gaussian_sigma_stop
            )
            task = create_gaussian_task(
                task_id, blocks_num, range(task_blocks_num), sigma
            )

        elif curve_distribution == LAPLACE:
            noise = random.uniform(
                self.config.laplace_noise_start, self.config.laplace_noise_stop
            )
            task = create_laplace_task(
                task_id, blocks_num, range(task_blocks_num), noise
            )
        elif curve_distribution == SUBSAMPLEGAUSSIAN:
            sigma = random.uniform(
                self.config.subsamplegaussian_sigma_start,
                self.config.subsamplegaussian_sigma_stop,
            )
            task = create_subsamplegaussian_task(
                task_id,
                blocks_num,
                range(task_blocks_num),
                self.config.subsamplegaussian_dataset_size,
                self.config.subsamplegaussian_batch_size,
                self.config.subsamplegaussian_epochs,
                sigma,
            )
        assert task is not None

        allocated_resources_event = self.env.event()
        # Wait till the demand-request has been delivered to the resource-manager
        yield self.resource_manager.task_demands_queue.put(
            (task, allocated_resources_event)
        )
        print("Task", task_id, "inserted demand")
        # Wait till the demand-request has been granted by the resource-manager
        yield allocated_resources_event

        print("Task ", task_id, "start running")
        # yield self.env.timeout()
        print("Task ", task_id, "completed at ", self.env.now)


class OnlineSimulator(BaseSimulator):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        env = simpy.rt.RealtimeEnvironment(factor=0.1, strict=False)
        resource_manager = ResourceManager(env, self.config)
        Tasks(env, resource_manager)
        env.run()
