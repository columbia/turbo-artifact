import simpy
from privacypacking.schedulers.methods import get_scheduler


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
        self.scheduler = get_scheduler(self.config)
        self.blocks_initialized = self.env.event()

        self.env.process(self.block_consumer())
        self.env.process(self.task_consumer())

    def block_consumer(self):
        def consume():
            block, generated_block_event = yield self.new_blocks_queue.get()
            self.scheduler.add_block(block)
            generated_block_event.succeed()

        # Consume all initial blocks
        initial_blocks_num = self.config.get_initial_blocks_num()
        for _ in range(initial_blocks_num):
            yield self.env.process(consume())
        self.blocks_initialized.succeed()

        while True:
            yield self.env.process(consume())
            if self.config.new_block_driven_scheduling:
                self.scheduler.schedule_queue()

    def task_consumer(self):
        scheduling_iteration = 0

        def consume():
            task_message = yield self.new_tasks_queue.get()
            return self.scheduler.add_task(task_message)

        # Consume initial tasks
        initial_tasks_num = self.config.get_initial_tasks_num()
        for _ in range(initial_tasks_num - 1):
            yield self.env.process(consume())

        while True:
            yield self.env.process(consume())
            # All schedulers support "new_task_driven_scheduling";
            # a new-task-arrival event triggers a new scheduling cycle
            if self.config.new_task_driven_scheduling:
                self.scheduler.schedule_queue()
                self.update_logs(scheduling_iteration)
                scheduling_iteration += 1

    def update_logs(self, scheduling_iteration):
        # TODO: improve the log period + perfs (it definitely is a bottleneck)
        if self.config.log_every_n_iterations and (
            (scheduling_iteration == 0)
            or (((scheduling_iteration + 1) % self.config.log_every_n_iterations) == 0)
        ):
            all_tasks = []
            for queue in self.scheduler.task_queue.values():
                all_tasks += queue.tasks
            self.config.logger.log(
                all_tasks + list(self.scheduler.tasks_info.allocated_tasks.values()),
                self.scheduler.blocks,
                list(self.scheduler.tasks_info.allocated_tasks.keys()),
                self.config,
            )
