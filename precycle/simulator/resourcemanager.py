import simpy
from loguru import logger
import time

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
        self.env.process(self.block_consumer())
        self.env.process(self.task_consumer())
        self.daemon_clock = self.env.process(self.daemon_clock())
        yield self.env.process(self.termination_clock())


    def termination_clock(self):
        yield self.block_production_terminated
        yield self.task_production_terminated
        self.daemon_clock.interrupt()
        yield self.task_consumption_terminated
        yield self.task_consumption_terminated
        logger.info(f"Terminating the simulation at {self.env.now}. Closing...")

    def daemon_clock(self):
        while True:
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

        logger.info("Done consuming blocks.")
        self.block_consumption_terminated.succeed()

    def task_consumer(self):
        def consume():
            task_message = yield self.new_tasks_queue.get()
            if isinstance(task_message, LastItem):
                return
            (task, allocated_resources_event) = task_message
            self.query_processor.try_run_task(task)
            allocated_resources_event.succeed()

        # Consume initial tasks
        for _ in range(self.config.tasks.initial_num):
            yield self.env.process(consume())

        logger.info("Done consuming initial tasks")

        while not self.task_production_terminated.triggered:
            yield self.env.process(consume())

        logger.info("Done consuming tasks")
        self.task_consumption_terminated.succeed()
