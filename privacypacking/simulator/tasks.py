from itertools import count
from loguru import logger


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
        initial_curves = self.config.get_initial_task_curves()
        for curve in initial_curves:
            self.env.process(self.task(next(self.task_count), curve))

        if self.config.task_arrival_frequency_enabled:
            while True:
                task_arrival_interval = self.config.set_task_arrival_time()
                self.env.process(self.task(next(self.task_count)))
                yield self.env.timeout(task_arrival_interval)

    def task(self, task_id: int, curve_distribution=None) -> None:
        """
        Task behavior. Sets its own demand, notifies resource manager of its existence,
        waits till it gets scheduled and then is executed
        """

        num_blocks = self.resource_manager.scheduler.safe_get_num_blocks()
        task = self.config.create_task(task_id, curve_distribution, num_blocks)

        logger.debug(f"Task: {task_id} generated at {self.env.now}")
        allocated_resources_event = self.env.event()
        yield self.resource_manager.new_tasks_queue.put(
            (task, allocated_resources_event)
        )

        yield allocated_resources_event
        logger.debug(f"Task: {task_id} scheduled at {self.env.now}")
