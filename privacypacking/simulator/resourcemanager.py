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
        self.scheduler = schedulers[self.config.scheduler_name]([], {}, self.config)

        self.blocks_initialized = self.env.event()

        self.env.process(self.block_consumer())
        self.env.process(self.task_consumer())

    def block_consumer(self):
        def consume():
            block, generated_block_event = yield self.new_blocks_queue.get()
            self.scheduler.safe_add_block(block)
            generated_block_event.succeed()

        # Consume all initial blocks
        initial_blocks_num = self.config.get_initial_blocks_num()
        for _ in range(initial_blocks_num):
            yield self.env.process(consume())
        self.blocks_initialized.succeed()
        # Keep consuming more incoming blocks
        while True:
            yield self.env.process(consume())

    def task_consumer(self):
        scheduling_iteration = 0
        waiting_events = {}

        def consume():
            task, allocated_resources_event = yield self.new_tasks_queue.get()
            waiting_events[task.id] = allocated_resources_event
            self.scheduler.add_task(task)

        # Consume all initial tasks
        initial_tasks_num = self.config.get_initial_tasks_num()
        for _ in range(initial_tasks_num):
            yield self.env.process(consume())
        # Keep consuming more incoming tasks
        while True:
            yield self.env.process(consume())
            # Schedule (it modifies the blocks) and update the list of pending tasks
            allocated_task_ids = self.scheduler.schedule()
            self.scheduler.update_allocated_tasks(allocated_task_ids)

            self.update_logs(scheduling_iteration)
            scheduling_iteration += 1

            # Wake-up all the tasks that have been scheduled
            for allocated_id in allocated_task_ids:
                waiting_events[allocated_id].succeed()
                del waiting_events[allocated_id]

    def update_logs(self, scheduling_iteration):
        # TODO: improve the log period + perfs (it definitely is a bottleneck)
        if self.config.log_every_n_iterations and (
            ((scheduling_iteration + 1) % self.config.log_every_n_iterations) == 0
        ):
            self.config.logger.log(
                self.scheduler.tasks + list(self.scheduler.allocated_tasks.values()),
                self.scheduler.blocks,
                list(self.scheduler.allocated_tasks.keys()),
                self.config,
            )
