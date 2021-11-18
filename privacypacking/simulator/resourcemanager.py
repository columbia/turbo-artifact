import simpy
from loguru import logger

from privacypacking.schedulers.methods import get_scheduler


class LastItem:
    def __init__(self):
        return


class ResourceManager:
    """
    Managing blocks and tasks arrival and schedules incoming tasks.
    """

    def __init__(self, environment, configuration):
        self.env = environment
        self.config = configuration
        self.block_arrival_interval = self.config.set_block_arrival_time()

        # To store the incoming tasks and blocks
        self.new_tasks_queue = simpy.Store(self.env)
        self.new_blocks_queue = simpy.Store(self.env)

        # Initialize the scheduler
        self.scheduler = get_scheduler(self.config, self.env)
        self.blocks_initialized = self.env.event()

        # Stopping conditions
        self.block_production_terminated = False
        self.task_production_terminated = False
        self.simulation_terminated = False

        # Start the processes
        self.env.process(self.block_consumer())
        self.env.process(self.task_consumer())
        self.env.process(self.daemon_clock())
        self.env.process(self.termination_clock())

    def termination_clock(self):

        # TODO: replace by an event?
        while not self.block_production_terminated:
            yield self.env.timeout(self.block_arrival_interval)

        logger.info(
            f"Block production terminated at {self.env.now}.\n Producing tasks for the last block..."
        )
        yield self.env.timeout(self.block_arrival_interval)

        logger.info(
            f"Task production terminated at {self.env.now}.\n Unlocking the remaining budget and allocating available tasks..."
        )
        self.task_production_terminated = True

        # We even wait a bit longer to ensure that all tasks are allocated (in case we need multiple scheduling steps)
        # TODO: add grace period that depends on T?
        yield self.env.timeout(self.config.scheduler_data_lifetime)

        logger.info(f"Terminating the simulation at {self.env.now}. Closing...")
        self.simulation_terminated = True

    def daemon_clock(self):
        while not self.simulation_terminated:
            yield self.env.timeout(1)
            logger.info(f"Simulation Time is: {self.env.now}")

    def block_consumer(self):
        def consume():
            item = yield self.new_blocks_queue.get()
            if isinstance(item, LastItem):
                return
            else:
                block, generated_block_event = item
                self.scheduler.add_block(block)
                generated_block_event.succeed()

        # Consume all initial blocks
        initial_blocks_num = self.config.get_initial_blocks_num()
        for _ in range(initial_blocks_num):
            yield self.env.process(consume())
        self.blocks_initialized.succeed()

        while not self.block_production_terminated:
            yield self.env.process(consume())
            # if self.config.new_block_driven_scheduling:
            #     self.scheduler.schedule_queue()

    def task_consumer(self):
        scheduling_iteration = 0

        def consume():
            task_message = yield self.new_tasks_queue.get()
            if isinstance(task_message, LastItem):
                return
            return self.scheduler.add_task(task_message)

        # Consume initial tasks
        initial_tasks_num = self.config.get_initial_tasks_num()
        for _ in range(initial_tasks_num - 1):
            yield self.env.process(consume())

        while not self.task_production_terminated:
            yield self.env.process(consume())

            # All schedulers support "new_task_driven_scheduling";
            # a new-task-arrival event triggers a new scheduling cycle
            # TODO: this is launching a new enless scheduling cycle in parallel? Why??
            if self.config.new_task_driven_scheduling:
                logger.warning("Running `scheduler.schedule_queue()`")
                self.scheduler.schedule_queue()
                self.update_logs(scheduling_iteration)
                scheduling_iteration += 1

        # Ask the scheduler to stop adding new scheduling steps
        self.scheduler.simulation_terminated = True

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
