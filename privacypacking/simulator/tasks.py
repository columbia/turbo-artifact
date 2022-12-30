import pandas as pd
from loguru import logger
from itertools import count
from privacypacking.utils.utils import REPO_ROOT
from privacypacking.simulator.resourcemanager import LastItem
from privacypacking.simulator.task_generator import (
    CSVTaskGenerator,
    PoissonTaskGenerator,
)


class Tasks:
    """Model task arrival rate and privacy demands."""

    def __init__(self, environment, resource_manager):
        self.env = environment
        self.resource_manager = resource_manager
        self.omegaconf = resource_manager.omegaconf
        self.task_count = count()

        self.tasks_path = REPO_ROOT.joinpath("data").joinpath(self.omegaconf.tasks.path)
        self.tasks_df = pd.read_csv(self.tasks_path)

        if "submit_time" in self.tasks_df:
            logger.info("Reading tasks in order with hardcoded arrival times.")
            self.omegaconf.tasks.initial_num = 0
            self.omegaconf.tasks.max_num = len(self.tasks_df)
            self.task_generator = CSVTaskGenerator(self.tasks_df)
        else:
            logger.info("Poisson sampling.")
            self.task_generator = PoissonTaskGenerator(
                self.tasks_df,
                self.omegaconf.tasks.avg_num_tasks_per_block,
                self.resource_manager.scheduler.blocks,
            )

        self.env.process(self.task_producer())

    def task_producer(self) -> None:
        """Generate tasks."""
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
                task_arrival_interval = (
                    self.task_generator.get_task_arrival_interval_time()
                )

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
        waits till it gets scheduled and then is executed.
        """

        task = self.task_generator.create_task(task_id)

        logger.debug(
            f"Task: {task_id} generated at {self.env.now}. Name: {task.name}. Blocks: {task.blocks}"
        )

        allocated_resources_event = self.env.event()
        yield self.resource_manager.new_tasks_queue.put(
            (task, allocated_resources_event)
        )

        yield allocated_resources_event
        logger.debug(f"Task: {task_id} scheduled at {self.env.now}")
