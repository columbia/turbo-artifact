import simpy.rt
from privacypacking.config import schedulers


class ResourceManager:
    """
    Managing blocks and tasks arrival and schedules incoming tasks.
    """

    def __init__(self, environment, configuration):
        self.env = environment
        self.config = configuration

        # To store the incoming tasks and blocks
        self.new_tasks_queue = simpy.Store(self.env)
        self.new_blocks_queue = simpy.Store(self.env)

        # Initialize the scheduler
        initial_tasks, initial_blocks = self.config.create_initial_tasks_and_blocks()
        self.scheduler = schedulers[self.config.scheduler_name](
            initial_tasks,
            initial_blocks,
        )
        self.initial_tasks_num = len(initial_tasks)
        self.initial_blocks_num = len(initial_tasks)

        self.env.process(self.block_consumer())
        self.env.process(self.task_consumer())

    def block_consumer(self):
        while True:
            # Pick the next block from the queue
            block, generated_block_event = yield self.new_blocks_queue.get()
            self.scheduler.safe_add_block(block)
            generated_block_event.succeed()

    def task_consumer(self):
        waiting_events = {}
        while True:
            # Pick the next task from the queue
            task, allocated_resources_event = yield self.new_tasks_queue.get()
            waiting_events[task.id] = allocated_resources_event
            # No synchronization needed for tasks as they are written/read sequentially
            self.scheduler.add_task(task)
            # Perform a 'schedule' step
            allocated_ids = self.scheduler.safe_schedule()
            self.update_logs()
            # Wake-up all the tasks that have been scheduled
            for allocated_id in allocated_ids:
                waiting_events[allocated_id].succeed()
                del waiting_events[allocated_id]

    def update_logs(self):
        pass
        # Update the logs for every time five new tasks arrive
        # TODO(later): check if this is a bottleneck (reserializing the whole log for just the last inputs)
        # if task.id == self.total_init_tasks - 1 or task.id % 5 == 0:
        #     self.config.logger.log(
        #         tasks + self.archived_allocated_tasks,
        #         self.blocks,
        #         allocated_ids
        #         + [
        #             allocated_task.id
        #             for allocated_task in self.archived_allocated_tasks
        #         ],
        #         self.config,
        #     )
        # logger.info(
        #     f"Scheduled tasks: {[t[0].id for t in waiting_tasks if t[0].id in allocated_ids]}"
        # )
