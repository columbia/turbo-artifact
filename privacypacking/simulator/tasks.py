from itertools import count
import random
from loguru import logger

from privacypacking.simulator.resourcemanager import LastItem
from privacypacking.budget import Task
from privacypacking.utils.utils import REPO_ROOT
from privacypacking.budget.task import UniformTask
from privacypacking.budget.block_selection import BlockSelectionPolicy
import pandas as pd


class Tasks:
    """
    Model task arrival rate and privacy demands.
    """

    def __init__(self, environment, resource_manager):
        self.env = environment
        self.resource_manager = resource_manager
        self.omegaconf = resource_manager.omegaconf
        self.task_count = count()

        self.tasks_path = REPO_ROOT.joinpath("data").joinpath(self.omegaconf.tasks.path)
        self.tasks = pd.read_csv(self.tasks_path)

        if self.omegaconf.tasks.sampling:
            logger.info("Poisson sampling.")
            # Uniform sampling with Poisson arrival from the CSV file
            def row_sampler(df):
                while True:  # Don't stop, `max_tasks` will take care of that
                    d = df.sample(1)
                    yield 0, d.squeeze()  # Same signature as iterrows()

            self.tasks_generator = row_sampler(self.tasks)
        else:
            logger.info("Reading tasks in order with hardcoded arrival times.")
            # Browse tasks in order with hardcoded arrival times
            self.tasks_generator = self.tasks.iterrows()
            self.omegaconf.tasks.max_num = len(self.tasks)
            self.omegaconf.tasks.initial_num = 0
            self.task_arrival_interval_generator = self.tasks[
                "relative_submit_time"
            ].iteritems()

        self.env.process(self.task_producer())

    def task_producer(self) -> None:
        """
        Generate tasks.
        """
        # Wait till blocks initialization is completed
        yield self.resource_manager.blocks_initialized

        # Produce initial tasks
        for _ in range(self.omegaconf.tasks.initial_num):
            self.env.process(self.task(next(self.task_count)))

        logger.debug("Done producing all the initial tasks.")

        if self.omegaconf.scheduler.method == "offline":
            return

        logger.debug(
            f"Generating online tasks now. Current count is: {self.task_count}"
        )
        while not self.resource_manager.task_production_terminated.triggered:
            task_id = next(self.task_count)
            if (
                self.omegaconf.tasks.max_num
                and task_id > self.omegaconf.tasks.max_num - 1
            ):
                # Send a special message to close the channel
                self.resource_manager.task_production_terminated.succeed()
                self.resource_manager.new_tasks_queue.put(LastItem())
                return
            else:
                task_arrival_interval = self.set_task_arrival_time()

                # No task can arrive after the end of the simulation
                # so we force them to appear right before the end of the last block
                task_arrival_interval = min(
                    task_arrival_interval,
                    self.omegaconf.blocks.max_num - self.env.now - 0.01,
                )
                self.env.process(self.task(task_id))
                yield self.env.timeout(task_arrival_interval)

        logger.info(
            f"Done generating tasks at time {self.env.now}. Current count is: {self.task_count}"
        )

    def task(self, task_id: int) -> None:
        """
        Task behavior. Sets its own demand, notifies resource manager of its existence,
        waits till it gets scheduled and then is executed
        """

        task = self.create_task(task_id)

        logger.debug(
            f"Task: {task_id} generated at {self.env.now}. Name: {task.name}. Blocks: {task.blocks}"
        )

        allocated_resources_event = self.env.event()
        yield self.resource_manager.new_tasks_queue.put(
            (task, allocated_resources_event)
        )

        yield allocated_resources_event
        logger.debug(f"Task: {task_id} scheduled at {self.env.now}")

    def create_task(self, task_id: int) -> Task:
        _, task_row = next(self.tasks_generator)

        # TODO: For now we read the utility/utility_beta from the config - one global utility applying to all tasks
        utility = self.omegaconf.tasks.utility
        utility_beta = self.omegaconf.tasks.utility_beta

        profit = 1 if "profit" not in task_row else float(task_row["profit"])
        name = task_id if "task_name" not in task_row else task_row["task_name"]

        task = UniformTask(
            id=task_id,
            query_id=int(task_row["query_id"]),
            query_type=task_row["query_type"],
            profit=profit,
            block_selection_policy=BlockSelectionPolicy.from_str(
                task_row["block_selection_policy"]
            ),
            n_blocks=int(task_row["n_blocks"]),
            utility=utility,
            utility_beta=utility_beta,
            name=name,
        )
        return task

    def set_task_arrival_time(self):
        if self.omegaconf.tasks.sampling == "poisson":
            task_arrival_interval = random.expovariate(
                self.omegaconf.tasks.avg_num_tasks_per_block
            )
        elif self.omegaconf.tasks.sampling == "constant":
            task_arrival_interval = self.task_arrival_interval

        else:
            _, task_arrival_interval = next(self.task_arrival_interval_generator)

        return task_arrival_interval
