from itertools import count

from loguru import logger

from privacypacking.simulator.resourcemanager import LastItem


class Tasks:
    """
    Model task arrival rate and privacy demands.
    """

    def __init__(self, environment, resource_manager):
        self.env = environment
        self.resource_manager = resource_manager
        self.config = resource_manager.config
        self.task_count = count()
        self.env.process(self.task_producer())

    def task_producer(self) -> None:
        """
        Generate tasks.
        """
        # Wait till blocks initialization is completed
        yield self.resource_manager.blocks_initialized

        # Produce initial tasks
        for _ in range(self.config.get_initial_tasks_num()):
            self.env.process(self.task(next(self.task_count)))

        logger.debug("Done producing all the initial tasks.")

        if self.config.omegaconf.scheduler.method == "offline":
            return

        logger.debug(
            f"Generating online tasks now. Current count is: {self.task_count}"
        )
        while not self.resource_manager.task_production_terminated.triggered:
            task_id = next(self.task_count)
            if self.config.max_tasks and task_id > self.config.max_tasks - 1:
                # Send a special message to close the channel
                self.resource_manager.task_production_terminated.succeed()
                self.resource_manager.new_tasks_queue.put(LastItem())
                return
            else:
                task_arrival_interval = self.config.set_task_arrival_time()

                # No task can arrive after the end of the simulation
                # so we force them to appear right before the end of the last block
                task_arrival_interval = min(
                    task_arrival_interval,
                    self.config.omegaconf.blocks.max_num - self.env.now - 0.01,
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

        task = self.config.create_task(task_id)

        logger.debug(
            f"Task: {task_id} generated at {self.env.now}. Name: {task.name}. Blocks: {list(task.budget_per_block.keys())}"
        )

        allocated_resources_event = self.env.event()
        yield self.resource_manager.new_tasks_queue.put(
            (task, allocated_resources_event)
        )

        yield allocated_resources_event
        logger.debug(f"Task: {task_id} scheduled at {self.env.now}")
