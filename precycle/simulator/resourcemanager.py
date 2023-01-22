import simpy
from loguru import logger


class LastItem:
    def __init__(self):
        return


class ResourceManager:
    """
    Managing blocks and tasks arrival and schedules incoming tasks.
    """

    def __init__(self, environment, db, budget_accountant, query_processor, config):
        self.env = environment
        self.config = config

        self.query_processor = query_processor
        self.budget_accountant = budget_accountant
        self.db = db

        # To store the incoming tasks and blocks
        self.new_tasks_queue = simpy.Store(self.env)
        self.new_blocks_queue = simpy.Store(self.env)

        self.blocks_initialized = self.env.event()

        # Stopping conditions
        self.block_production_terminated = self.env.event()
        self.task_production_terminated = self.env.event()
        self.simulation_terminated = self.env.event()

    def terminate_simulation(self):
        # TODO: Maybe a bit brutal, if it causes problems we should use events (like above)
        # self.scheduling.interrupt()
        self.daemon_clock.interrupt()

    def start(self):
        # Start the processes
        self.env.process(self.block_consumer())
        task_consumed_event = self.env.process(self.task_consumer())


        self.daemon_clock = self.env.process(self.daemon_clock())
        self.env.process(self.termination_clock())


    def termination_clock(self):
        # Wait for all the blocks to be produced before moving on
        yield self.block_production_terminated
        logger.info(
            f"Block production terminated at {self.env.now}.\n Producing tasks for the last block..."
        )

        yield self.env.timeout(self.config.blocks.arrival_interval)

        # All the blocks are here, time to stop creating online tasks
        if not self.task_production_terminated.triggered:
            self.task_production_terminated.succeed()
            logger.info(
                f"Task production terminated at {self.env.now}.\n Unlocking the remaining budget and allocating available tasks..."
            )

        # # We even wait a bit longer to ensure that all tasks are allocated (in case we need multiple scheduling steps)
        # # TODO: add grace period that depends on T?
        # yield self.env.timeout(self.config.scheduler.data_lifetime)

        logger.info(f"Terminating the simulation at {self.env.now}. Closing...")

    def daemon_clock(self):
        while not self.simulation_terminated.triggered:
            try:
                yield self.env.timeout(1)
                logger.info(f"Simulation Time is: {self.env.now}")
            except simpy.Interrupt as i:
                return

    def block_consumer(self):
        def consume():
            item = yield self.new_blocks_queue.get()
            block_id, generated_block_event = item
            
            block_data_path = self.config.blocks.block_data_path + f"/block_{block_id}.csv"
            self.db.add_new_block(block_data_path)
            self.budget_accountant.add_new_block_budget()

            generated_block_event.succeed()

        # Consume all initial blocks
        for _ in range(self.config.blocks.initial_num):
            yield self.env.process(consume())
        self.blocks_initialized.succeed()
        logger.info(f"Initial blocks: {self.budget_accountant.get_blocks_count()}")

        while not self.block_production_terminated.triggered:
            yield self.env.process(consume())

        logger.info("Done producing blocks.")

    def task_consumer(self):
        def consume():
            task_message = yield self.new_tasks_queue.get()
            if isinstance(task_message, LastItem):
                return
            return self.query_processor.try_run_task(task_message)

        # Consume initial tasks
        for _ in range(self.config.tasks.initial_num):
            yield self.env.process(consume())

        logger.info("Done consuming initial tasks")

        while not self.task_production_terminated.triggered:
            yield self.env.process(consume())

        logger.info("Done consuming tasks")

        # Ask the scheduler to stop adding new scheduling steps
        # self.terminate_simulation()

        # TODO: just replace by `task_consumed_event`?
        self.simulation_terminated.succeed()
