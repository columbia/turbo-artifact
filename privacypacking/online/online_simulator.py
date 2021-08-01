"""
Model a privacy-resource-manager that grants privacy-budget resources to incoming tasks
according to a given scheduling mechanism.

ResourceManager has several block-resources each one of them having a privacy budget-capacity.
Incoming tasks arrive each one of them having a privacy budget-demand for one or more blocks.
Resources are non-replenishable.

ResourceManager owns a scheduling mechanism for servicing tasks according to a given policy.
"""

from itertools import count

import simpy.rt
from privacypacking.base_simulator import BaseSimulator
from privacypacking.budget.block import create_block
from privacypacking.budget.task import create_gaussian_task
from privacypacking.online.schedulers import fcfs, dpf
from privacypacking.utils.utils import *

schedulers = {
    FCFS: fcfs.FCFS,
    DPF: dpf.DPF
}


class ResourceManager:
    """
    A resource-manager has several blocks each one of them having a privacy-budget.
    While privacy-budgets are not replenishable in the sense that they can't be returned after used
    by a task more privacy budget may be added to them or additional blocks with privacy-budgets may arrive.

    The resource-manager has a traffic generator process that causes tasks to arrive and be granted resources.

    As a task consumes privacy budget resources the level of those resources goes down.
    A task must be granted "all" the resources that it demands or "nothing".
    """

    def __init__(self, env):
        self.env = env
        self.config = {}

        self.num_blocks = self.config.get('num_blocks', 1)
        self.alphas = self.config.get('alphas', [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64])
        self.renyi_epsilon = self.config.get('renyi_epsilon', 10)
        self.renyi_delta = self.config.get('renyi_delta', 0.01)
        self.renyi_delta = self.config.get('renyi_delta', 0.01)
        self.blocks_num = self.config.get('blocks_num', 1)
        self.task_arrival_interval = self.config.get('task_arrival_interval', 3)
        self.scheduler = schedulers[self.config.get('scheduler', FCFS)]

        self.blocks = []

        # To store the incoming task demands
        self.task_demands_queue = simpy.Store(self.env)

        # A ResourceManager has two persistent processes.
        # One that models the arrival of new resources
        # Create blocks statically for now
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
        self.blocks = [create_block(next(block_id), self.renyi_epsilon, self.renyi_delta)
                       for _ in range(self.blocks_num)]
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
            s = self.scheduler([t[0] for t in waiting_tasks], self.blocks)
            scheduled_tasks_idxs = s.schedule()
            print("Scheduled tasks", [waiting_tasks[i][0].id for i in scheduled_tasks_idxs])

            # Wake-up all the tasks that have been scheduled
            for i in scheduled_tasks_idxs:
                waiting_tasks[i][1].succeed()

            # todo: resolve race-condition between task-demands/budget updates and blocks; perhaps use mutex for quicker solution

            # Update/consume block-budgets
            # todo: lock

            for i in scheduled_tasks_idxs:
                t = waiting_tasks[i][0]
                for block_id in t.block_ids:
                    block_demand_budget = t.budget_per_block[block_id]
                    # Get block with "block_id"
                    block = get_block_by_block_id(self.blocks, block_id)
                    block.budget -= block_demand_budget

            # Delete the tasks that have been scheduled from the waiting list
            waiting_tasks = [t for i, t in enumerate(waiting_tasks)
                             if i not in scheduled_tasks_idxs]


class Tasks:
    """
    Model task arrival rate and privacy demands.
    Each task's arrival time, privacy demands is determined by configuration.
    A new process is spawned for each task.
    """

    def __init__(self, env, resource_manager):
        self.env = env
        self.resource_manager = resource_manager
        env.process(self.generate_tasks())

    def generate_tasks(self):
        """
        Generate tasks.
        Various configuration parameters determine the distribution of task
        arrival times as well as the demands of each task.
        """

        task_id = count()
        arrival_interval_dist = 10  # self.env.rand.expovariate, 1 / self.env.config['task.arrival_interval']

        while True:
            self.env.process(self.task(next(task_id)))
            yield self.env.timeout(arrival_interval_dist)

    def task(self, task_id):
        """Generated task behavior."""

        print("Generated task: ", task_id)
        # Determine demands
        # sigmas = np.linspace(0.1, 1, 10)   # specify distribution, parameters, num_of_blocks, range
        s = 0.1
        blocks_num = 1
        task = create_gaussian_task(task_id, blocks_num, range(blocks_num), s)

        allocated_resources_event = self.env.event()
        # Wait till the demand-request has been delivered to the resource-manager
        yield self.resource_manager.task_demands_queue.put((task, allocated_resources_event))
        print("Task", task_id, "inserted demand")
        # Wait till the demand-request has been granted by the resource-manager
        yield allocated_resources_event

        # yield self.active.put(1)
        print('Task ', task_id, 'start running')
        # yield self.env.timeout()
        print('Task ', task_id, 'completed at ', self.env.now)


class OnlineSimulator(BaseSimulator):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        env = simpy.rt.RealtimeEnvironment(factor=0.1)
        resource_manager = ResourceManager(env)
        tasks = Tasks(env, resource_manager)
        env.run()


if __name__ == '__main__':
    env = simpy.rt.RealtimeEnvironment(factor=0.1)
    rm = ResourceManager(env)
    ts = Tasks(env, rm)
    env.run()
