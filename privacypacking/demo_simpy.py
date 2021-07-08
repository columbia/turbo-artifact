from typing import Generator

import simpy
from loguru import logger
from simpy import Environment, Process, Timeout
from simpy.events import Event


def car(env: Environment):
    while True:
        logger.info("Start parking at %d" % env.now)
        parking_duration = 5
        yield env.timeout(parking_duration)

        logger.info("Start driving at %d" % env.now)
        trip_duration = 2
        yield env.timeout(trip_duration)


class Car(object):
    def __init__(self, env):
        self.env = env
        self.action: Process = env.process(self.run())

    def run(self) -> Generator[Event, None, None]:
        while True:
            print("Start parking and charging at %d" % self.env.now)
            charge_duration = 5
            # We may get interrupted while charging the battery
            try:
                yield self.env.process(self.charge(charge_duration))
            except simpy.Interrupt:
                # When we received an interrupt, we stop charging and
                # switch to the "driving" state
                print("Was interrupted. Hope, the battery is full enough ...")
            print("Start driving at %d" % self.env.now)
            trip_duration = 2
            yield self.env.timeout(trip_duration)

    def charge(self, duration):
        yield self.env.timeout(duration)


def run_function():
    env = Environment()
    car_process = env.process(car(env))
    env.run(until=15)
    logger.info(car_process)


def run_class():
    env = Environment()
    car = Car(env)

    def driver(env, car):
        yield env.timeout(1)
        car.action.interrupt()

    env.process(driver(env, car))

    env.run(until=15)


if __name__ == "__main__":
    run_class()
