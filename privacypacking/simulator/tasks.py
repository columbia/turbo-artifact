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

        if self.config.omegaconf.scheduler.method != "offline":
            while not self.resource_manager.task_production_terminated:
                task_arrival_interval = self.config.set_task_arrival_time()
                task_id = next(self.task_count)
                self.env.process(self.task(task_id))
                yield self.env.timeout(task_arrival_interval)

                # Todo: Is max-tasks ever not-None anymore?
                if (
                    self.config.max_tasks is not None
                    and task_id == self.config.max_tasks - 1
                ):
                    self.resource_manager.task_production_terminated = True

            # Send a special message to close the channel
            self.resource_manager.new_tasks_queue.put(LastItem())

    def task(self, task_id: int) -> None:
        """
        Task behavior. Sets its own demand, notifies resource manager of its existence,
        waits till it gets scheduled and then is executed
        """

        task = self.config.create_task(task_id)

        # logger.debug(
        #     f"Task: {task_id} generated at {self.env.now}. Name: {task.name}. Blocks: {list(task.budget_per_block.keys())}"
        # )

        allocated_resources_event = self.env.event()
        yield self.resource_manager.new_tasks_queue.put(
            (task, allocated_resources_event)
        )

        yield allocated_resources_event
        # logger.debug(f"Task: {task_id} scheduled at {self.env.now}")
