from loguru import logger
from itertools import count


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

    def task_producer(self):
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

    def task(self, task_id, curve_distribution=None):
        """
        Task behavior. Sets its own demand, notifies resource manager of its existence,
        waits till it gets scheduled and then is executed
        """
        logger.debug(f"Task: {task_id} generated at {self.env.now}")
        curve_distribution = (
            self.config.set_curve_distribution()
            if curve_distribution is None
            else curve_distribution
        )

        num_blocks = self.resource_manager.scheduler.safe_get_num_blocks()
        task_blocks_num = self.config.set_task_num_blocks(
            curve_distribution, num_blocks
        )
        policy = self.config.get_policy(curve_distribution)
        # Ask the stateful scheduler to set the block ids of the task according to the policy function
        selected_block_ids = self.resource_manager.scheduler.safe_select_block_ids(
            task_blocks_num, policy
        )

        task = self.config.create_task(task_id, curve_distribution, selected_block_ids)
        allocated_resources_event = self.env.event()

        yield self.resource_manager.new_tasks_queue.put(
            (task, allocated_resources_event)
        )

        yield allocated_resources_event
        # logger.debug(f"Task: {task_id} completed at {self.env.now}")
