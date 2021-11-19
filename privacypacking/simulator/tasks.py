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
        initial_curves = self.config.get_initial_task_curves()
        for curve in initial_curves:
            self.env.process(self.task(next(self.task_count), curve))

        if self.config.task_arrival_frequency_enabled:
            while not self.resource_manager.task_production_terminated:
                task_arrival_interval = self.config.set_task_arrival_time()
                task_id = next(self.task_count)
                self.env.process(self.task(task_id))
                yield self.env.timeout(task_arrival_interval)

                if (
                    self.config.max_tasks is not None
                    and task_id == self.config.max_tasks - 1
                ):
                    self.resource_manager.task_production_terminated = True

            # Send a special message to close the channel
            self.resource_manager.new_tasks_queue.put(LastItem())

    def task(self, task_id: int, curve_distribution=None) -> None:
        """
        Task behavior. Sets its own demand, notifies resource manager of its existence,
        waits till it gets scheduled and then is executed
        """

        num_blocks = self.resource_manager.scheduler.get_num_blocks()
        task = self.config.create_task(task_id, curve_distribution, num_blocks)

        logger.debug(f"Task: {task_id} generated at {self.env.now}")
        allocated_resources_event = self.env.event()
        yield self.resource_manager.new_tasks_queue.put(
            (task, allocated_resources_event)
        )

        yield allocated_resources_event
        logger.debug(f"Task: {task_id} scheduled at {self.env.now}")
