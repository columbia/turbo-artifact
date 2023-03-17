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

        # # To store the incoming tasks and blocks
        self.new_tasks_queue = simpy.Store(self.env)
        self.new_blocks_queue = simpy.Store(self.env)

        self.blocks_initialized = self.env.event()

        # Stopping conditions
        self.block_production_terminated = self.env.event()
        self.task_production_terminated = self.env.event()
        self.block_consumption_terminated = self.env.event()
        self.task_consumption_terminated = self.env.event()

    def start(self):
        self.daemon_clock = self.env.process(self.daemon_clock())

        self.env.process(self.block_consumer())
        self.env.process(self.task_consumer())

        # Termination conditions
        yield self.block_production_terminated
        yield self.task_production_terminated
        yield self.block_consumption_terminated
        yield self.task_consumption_terminated
        self.daemon_clock.interrupt()
        logger.info(f"Terminating the simulation at {self.env.now}. Closing...")

    def daemon_clock(self):
        while True:
            try:
                yield self.env.timeout(1)
                logger.info(f"Simulation Time is: {self.env.now}")
            except simpy.Interrupt as i:
                return

    def block_consumer(self):
        while True:
            block_message = yield self.new_blocks_queue.get()

            if isinstance(block_message, LastItem):
                logger.info("Done consuming blocks.")
                self.block_consumption_terminated.succeed()
                return

            block_id = block_message
            block_data_path = (
                self.config.blocks.block_data_path + f"/block_{block_id}"
            )
            self.db.add_new_block(block_data_path)
            self.budget_accountant.add_new_block_budget()

            if self.config.blocks.initial_num == block_id + 1:
                self.blocks_initialized.succeed()

    def task_consumer(self):
        while True:
            task_message = yield self.new_tasks_queue.get()

            if isinstance(task_message, LastItem):
                logger.info("Done consuming tasks")
                self.task_consumption_terminated.succeed()
                return

            task = task_message
            self.query_processor.try_run_task(task)
