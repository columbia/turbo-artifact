import time

import simpy
from loguru import logger


class ToySimulator:
    """Just a smoke test/pedagogical class to show that the simulator has no race condition.
    To see the logs, run: ` pytest -s tests/test_discrete_simulator.py`
    """

    def __init__(self, env) -> None:
        self.env = env
        self.main_process = self.env.process(self.main())
        self.new_block_event = self.env.event()
        self.new_task_event = self.env.event()

        self.env.process(self.block_gen())
        self.env.process(self.task_gen())

        self.results = []

    def main(self):
        logger.info("Starting the simulator")

        while True:
            logger.info("Waiting for blocks")
            yield self.new_block_event | self.new_task_event
            logger.info(
                f"Got a block! Or a task! Current simulation time: {self.env.now}"
            )
            self.results.append(self.env.now)

            time.sleep(2)
            logger.info(
                f"Done computing the complex scheduling. Current simulation time: {self.env.now}"
            )
            self.results.append(self.env.now)

    def block_gen(self):
        logger.info("Starting the block generator.")
        while True:
            yield self.env.timeout(3)
            logger.info(
                f"The generator got a block! Current simulation time: {self.env.now}"
            )
            self.new_block_event.succeed()
            logger.info("Back to the block gen loop")
            self.new_block_event = self.env.event()

    def task_gen(self):
        logger.info("Starting the task generator.")
        while True:
            yield self.env.timeout(3)
            logger.info(
                f"The task generator is done sleeping. Time to interrupt! Current simulation time: {self.env.now}"
            )
            self.new_task_event.succeed()
            logger.info("Back to the task gen loop")
            self.new_task_event = self.env.event()


def test_discrete_simulator():
    env = simpy.Environment()
    sim = ToySimulator(env)
    env.run(until=7)

    assert sim.results == [3, 3, 6, 6]
