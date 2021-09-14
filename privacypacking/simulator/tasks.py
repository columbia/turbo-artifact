from itertools import count
from pathlib import Path
from loguru import logger


class Tasks:
    """
    Model task arrival rate and privacy demands.
    """

    def __init__(self, environment, resource_manager):
        self.env = environment
        self.resource_manager = resource_manager
        self.config = resource_manager.config
        self.task_count = count()
        self.env.process(self.task_producer())

    def task_producer(self) -> None:
        """
        Generate tasks.
        """
        # Wait till blocks initialization is completed
        yield self.resource_manager.blocks_initialized

        # Produce initial tasks
        initial_curves = self.config.get_initial_task_curves()
        for curve in initial_curves:
            self.env.process(self.task(next(self.task_count), curve))

        if self.config.task_arrival_frequency_enabled:
            while True:
                task_arrival_interval = self.config.set_task_arrival_time()
                self.env.process(self.task(next(self.task_count)))
                yield self.env.timeout(task_arrival_interval)

    def task(self, task_id: int, curve_distribution=None) -> None:
        """
        Task behavior. Sets its own demand, notifies resource manager of its existence,
        waits till it gets scheduled and then is executed
        """

        num_blocks = self.resource_manager.scheduler.safe_get_num_blocks()
        task = self.config.create_task(task_id, curve_distribution, num_blocks)

        logger.debug(f"Task: {task_id} generated at {self.env.now}")
        allocated_resources_event = self.env.event()
        yield self.resource_manager.new_tasks_queue.put(
            (task, allocated_resources_event)
        )

        yield allocated_resources_event
        # logger.debug(f"Task: {task_id} completed at {self.env.now}")


# TODO: Build more custom task generators if the simulator works.
# For now (offline case), it is easier to just change the function that generates the initial tasks.

# class TasksFromFile(Tasks):
#     """Loads the tasks (budget curves and number of blocks) from arbitrary yaml files,
#     instead of generating them from a few configuration parameters"""

#     # TODO: Add an optional profit field to the files
#     # TODO: Add optional frequencies to the files
#     # TODO: Add optional policy to each task (not tied to the budget)

#     def __init__(self, environment, resource_manager, blocks_and_budgets_path: Path):
#         self.blocks_and_budgets = load_blocks_and_budgets_from_dir(
#             blocks_and_budgets_path
#         )
#         super().__init__(environment, resource_manager)

#     def task_producer(self):
#         """
#         Generate tasks.
#         """
#         # Wait till blocks initialization is completed
#         yield self.resource_manager.blocks_initialized

#         # Produce initial tasks from the file (this is the only difference with the base class)
#         # initial_curves = self.config.get_initial_task_curves()
#         # for curve in initial_curves:
#         #     self.env.process(self.task(next(self.task_count), curve))
#         for _ in range(self.config.config["tasks_spec"]["initial_num"]):
#             self.env.process(self.task(next(self.task_count)))

#         if self.config.task_arrival_frequency_enabled:
#             while True:
#                 task_arrival_interval = self.config.set_task_arrival_time()
#                 self.env.process(self.task(next(self.task_count)))
#                 yield self.env.timeout(task_arrival_interval)

#     def task(self, task_id, curve_distribution=None):

#         task_blocks_num, budget = random.choice(self.blocks_and_budgets)

#         num_blocks = self.resource_manager.scheduler.safe_get_num_blocks()

#         PROFIT = 1
#         POLICY = Random

#         # Ask the stateful scheduler to set the block ids of the task according to the policy function
#         selected_block_ids = self.resource_manager.scheduler.safe_select_block_ids(
#             task_blocks_num, POLICY
#         )

#         logger.debug(list(selected_block_ids))

#         task = UniformTask(
#             id=task_id,
#             profit=PROFIT,
#             block_ids=selected_block_ids,
#             budget=budget,
#         )
#         allocated_resources_event = self.env.event()

#         yield self.resource_manager.new_tasks_queue.put(
#             (task, allocated_resources_event)
#         )

#         yield allocated_resources_event
