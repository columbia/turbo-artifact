import pandas as pd
from loguru import logger
from itertools import count
from precycle.utils.utils import REPO_ROOT
from precycle.simulator.resourcemanager import LastItem
from precycle.simulator.task_generator import (
    CSVTaskGenerator,
    PoissonTaskGenerator,
)


class Tasks:
    """Model task arrival rate and privacy demands."""

    def __init__(self, environment, resource_manager):
        self.env = environment
        self.resource_manager = resource_manager
        self.config = resource_manager.config
        self.task_count = count()

        self.tasks_path = REPO_ROOT.joinpath("data").joinpath(self.config.tasks.path)
        self.tasks_df = pd.read_csv(self.tasks_path)

        if "submit_time" in self.tasks_df:
            logger.info("Reading tasks in order with hardcoded arrival times.")
            self.config.tasks.initial_num = 0
            self.config.tasks.max_num = len(self.tasks_df)
            self.task_generator = CSVTaskGenerator(self.tasks_df, self.config)
        else:
            logger.info("Poisson sampling.")
            self.task_generator = PoissonTaskGenerator(
                self.tasks_df,
                self.config.tasks.avg_num_tasks_per_block,
                self.config
            )

        self.env.process(self.task_producer())

    def task_producer(self) -> None:
        """Generate tasks."""
        # Wait till blocks initialization is completed
        yield self.resource_manager.blocks_initialized

        # Produce initial tasks
        for _ in range(self.config.tasks.initial_num):
            self.env.process(self.task(next(self.task_count)))

        logger.debug("Done producing all the initial tasks.")
        logger.debug(
            f"Generating online tasks now. Current count is: {self.task_count}"
        )
        while not self.resource_manager.task_production_terminated.triggered:
            task_id = next(self.task_count)
            if (
                self.config.tasks.max_num
                and task_id > self.config.tasks.max_num - 1
            ) or (
                not self.config.tasks.max_num
                and self.resource_manager.block_production_terminated.triggered
            ):
                # Termination condition: either tasks max num has been reached
                # or there is no tasks max num limit and the blocks max num has been reached
                # Send a special message to close the channel
                self.resource_manager.task_production_terminated.succeed()
                self.resource_manager.new_tasks_queue.put(LastItem())
                break

            task_arrival_interval = self.task_generator.get_task_arrival_interval_time()

            # No task can arrive after the end of the simulation
            # so we force them to appear right before the end of the last block
            task_arrival_interval = min(
                task_arrival_interval,
                self.config.blocks.max_num - self.env.now - 0.01,
            )

            self.env.process(self.task(task_id))
            yield self.env.timeout(task_arrival_interval)

        logger.info(
            f"Done generating tasks at time {self.env.now}. Current count is: {self.task_count}"
        )

    def task(self, task_id: int) -> None:
        """
        Task behavior. Sets its own demand, notifies resource manager of its existence,
        waits till it gets scheduled and then is executed.
        """
        blocks_count = self.resource_manager.budget_accountant.get_blocks_count()
        task = self.task_generator.create_task(task_id, blocks_count)

        logger.debug(
            f"Task: {task_id} generated at {self.env.now}. Name: {task.name}. Blocks: {task.blocks}"
        )

        allocated_resources_event = self.env.event()
        yield self.resource_manager.new_tasks_queue.put(
            (task, allocated_resources_event)
        )

        yield allocated_resources_event
        logger.debug(f"Task: {task_id} scheduled at {self.env.now}")
