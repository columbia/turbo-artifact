import simpy
from loguru import logger
from privacypacking.schedulers.methods import initialize_scheduler


class LastItem:
    def __init__(self):
        return


class ResourceManager:
    """
    Managing blocks and tasks arrival and schedules incoming tasks.
    """

    def __init__(self, environment, omegaconf):
        self.env = environment
        self.omegaconf = omegaconf

        # To store the incoming tasks and blocks
        self.new_tasks_queue = simpy.Store(self.env)
        self.new_blocks_queue = simpy.Store(self.env)

        # Initialize the scheduler
        # self.scheduler = initialize_scheduler(omegaconf, self.env)
        self.blocks_initialized = self.env.event()

        # Stopping conditions
        self.block_production_terminated = self.env.event()
        self.task_production_terminated = self.env.event()
        self.simulation_terminated = self.env.event()

    def terminate_simulation(self):
        # TODO: Maybe a bit brutal, if it causes problems we should use events (like above)
        self.scheduling.interrupt()
        self.daemon_clock.interrupt()

    def start(self):
        # Start the processes
        self.env.process(self.block_consumer())
        task_consumed_event = self.env.process(self.task_consumer())

        # In the online case, start a clock to terminate the simulation
        if self.omegaconf.scheduler.method == "batch":
            self.daemon_clock = self.env.process(self.daemon_clock())
            self.env.process(self.termination_clock())

            self.scheduling = self.env.process(
                self.scheduler.run_batch_scheduling(
                    simulation_termination_event=self.simulation_terminated,
                    period=self.omegaconf.scheduler.scheduling_wait_time,
                )
            )

        # elif self.omegaconf.scheduler.method == "offline":
        #     self.block_production_terminated.succeed()
        #     self.task_production_terminated.succeed()

        #     yield task_consumed_event
        #     logger.info(
        #         "The scheduler has consumed all the tasks. Time to allocate them."
        #     )
        #     self.scheduler.schedule_queue()

    def termination_clock(self):
        # Wait for all the blocks to be produced before moving on
        yield self.block_production_terminated
        logger.info(
            f"Block production terminated at {self.env.now}.\n Producing tasks for the last block..."
        )

        yield self.env.timeout(self.omegaconf.blocks.arrival_interval)

        # All the blocks are here, time to stop creating online tasks
        if not self.task_production_terminated.triggered:
            self.task_production_terminated.succeed()
            logger.info(
                f"Task production terminated at {self.env.now}.\n Unlocking the remaining budget and allocating available tasks..."
            )

        # We even wait a bit longer to ensure that all tasks are allocated (in case we need multiple scheduling steps)
        # TODO: add grace period that depends on T?
        yield self.env.timeout(self.omegaconf.scheduler.data_lifetime)

        logger.info(f"Terminating the simulation at {self.env.now}. Closing...")
        # self.terminate_simulation()
        # self.simulation_terminated.succeed()

    def daemon_clock(self):
        while not self.simulation_terminated.triggered:
            try:
                yield self.env.timeout(1)
                logger.info(f"Simulation Time is: {self.env.now}")
            except simpy.Interrupt as i:
                return

    def block_consumer(self):
        # Needlessly convoluted?
        def consume():
            item = yield self.new_blocks_queue.get()
            block, generated_block_event = item
            self.scheduler.add_block(block)
            generated_block_event.succeed()

        # Consume all initial blocks
        for _ in range(self.omegaconf.blocks.initial_num):
            yield self.env.process(consume())
        self.blocks_initialized.succeed()
        logger.info(f"Initial blocks: {len(self.scheduler.blocks)}")

        while not self.block_production_terminated.triggered:
            yield self.env.process(consume())

        logger.info("Done producing blocks.")

    def task_consumer(self):
        def consume():
            task_message = yield self.new_tasks_queue.get()
            if isinstance(task_message, LastItem):
                return
            return self.scheduler.add_task(task_message)

        # Consume initial tasks
        for _ in range(self.omegaconf.tasks.initial_num):
            yield self.env.process(consume())

        logger.info("Done consuming initial tasks")

        while not self.task_production_terminated.triggered:
            yield self.env.process(consume())

        logger.info("Done consuming tasks")

        # Ask the scheduler to stop adding new scheduling steps
        # self.terminate_simulation()

        # TODO: just replace by `task_consumed_event`?
        self.simulation_terminated.succeed()
