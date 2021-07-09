# Goal: write a tiny prototype of stateful scheduler
#       with dynamic resources + jobs
#       with the most simple architecture possible (no Resources?)
#       and ideally a black-box scheduler

from os import name
from typing import Dict, Generator, List

from loguru import logger
from simpy import Environment, Process, Timeout
from simpy.events import Event


class Resource:
    pass


class Request:
    pass


class Job:
    def __init__(self, env: Environment, delay, request: Request) -> None:
        self.env = env
        self.delay = delay
        self.request = request

        self.process: Process = env.process(self.process_generator())
        self.allocated: Event = env.event()

    def process_generator(self) -> Generator[Event, None, None]:
        # Wait for `delay` steps to start
        # TODO: separate job spawner

        yield self.env.timeout(self.delay)
        logger.info("Starting to wait for resource.")
        # Wait for `allocated` to succeed
        # TODO: add timeout (|)
        yield self.allocated
        logger.info("Allocated!")

        # Ask for some resource. How to ping the scheduler? Interrupt?

        # Yield is_allocated -> and the scheduler will trigger the event
        # https://simpy.readthedocs.io/en/latest/topical_guides/events.html#example-usages-for-event


class Scheduler:
    def __init__(self, env: Environment, resources, job_list: List[Job]) -> None:
        # TODO: Scheduler is not a process, no env?
        self.env = env
        self.resources = resources
        self.job_list = job_list

    def run(self):
        while True:
            allocatable_jobs = list(filter(self.allocatable, self.job_list))
            logger.info(f"Allocatable jobs: {allocatable_jobs}")

            if not allocatable_jobs:
                # TODO: yield to wait for new jobs or resources
                break

            # TODO: sort jobs too (DPF style for instance)
            allocated_job = allocatable_jobs.pop()
            request = allocated_job.request
            # TODO: all-in-one function, like self.allocate_request(job)?
            self.update_resources(request)

            # TODO: how to incorporate dynamic resources too? Interrupt, or yield?

            # Trigger the job
            allocated_job.allocated.succeed()
            # Don't do anything else with the rest of the allocatable jobs: the state might have changed

    def update_resources(self, request):
        # TODO: remove the consumed resource
        logger.debug("Updating resources")
        self.resources = self.resources

    def allocatable(self, job: Job) -> bool:
        logger.debug(f"Comparing request: {job.request} to resources: {self.resources}")
        # TODO: call a fancy policy here
        return not job.allocated.triggered


if __name__ == "__main__":
    env = Environment()
    job_1 = Job(env, delay=0, request={"1": 1})
    job_2 = Job(env, delay=0, request={"2": 1})
    resources = {
        "1": 1,
        "2": 1,
    }
    scheduler = Scheduler(env, resources, [job_1, job_2])

    scheduler.run()
