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
            self.config
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
        scheduling_iteration = 0
        waiting_events = {}
        print(self.scheduler.tasks)
        while True:
            # Pick the next task from the queue
            task, allocated_resources_event = yield self.new_tasks_queue.get()
            print(task.id)
            waiting_events[task.id] = allocated_resources_event
            # No synchronization needed for tasks as they are written/read sequentially
            self.scheduler.add_task(task)

            # Schedule (it modifies the blocks) and update the list of pending tasks
            allocated_task_ids = self.scheduler.schedule()
            print(allocated_task_ids)
            self.scheduler.update_allocated_tasks(allocated_task_ids)

            self.update_logs(scheduling_iteration)
            scheduling_iteration += 1

            # Wake-up all the tasks that have been scheduled
            for allocated_id in allocated_task_ids:
                waiting_events[allocated_id].succeed()
                del waiting_events[allocated_id]

    def update_logs(self, scheduling_iteration):
        # TODO: improve the log period + perfs
        if self.config.log_every_n_iterations and (
            ((scheduling_iteration + 1) % self.config.log_every_n_iterations) == 0
        ):
            print(list(self.scheduler.allocated_tasks.keys()))
            self.config.logger.log(
                self.scheduler.tasks + list(self.scheduler.allocated_tasks.values()),
                self.scheduler.blocks,
                list(self.scheduler.allocated_tasks.keys()),
                self.config,
            )
