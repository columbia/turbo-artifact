import random
from loguru import logger
from precycle.task import Task


class TaskGenerator:
    def __init__(self, df_tasks, config) -> None:
        self.config = config
        self.tasks = df_tasks

    def sample_task_row(self):
        raise NotImplementedError("Must override")

    def create_task(self, task_id, num_blocks):
        task_row = self.sample_task_row(num_blocks).squeeze()

        query_id = int(task_row["query_id"])
        name = task_id if "task_name" not in task_row else task_row["task_name"]
        num_requested_blocks = int(task_row["n_blocks"])
        requested_blocks = (num_blocks - num_requested_blocks, num_blocks - 1)

        task = Task(
            id=task_id,
            query_id=query_id,
            query_type=task_row["query_type"],
            query=eval(task_row["query"]),
            blocks=requested_blocks,
            n_blocks=num_requested_blocks,
            utility=float(task_row["utility"]),
            utility_beta=float(task_row["utility_beta"]),
            name=name,
        )
        # print("\nTask", task.dump())
        return task


class PoissonTaskGenerator(TaskGenerator):
    def __init__(self, df_tasks, avg_num_tasks_per_block, config) -> None:
        super().__init__(df_tasks, config)
        self.avg_num_tasks_per_block = avg_num_tasks_per_block

    def sample_task_row(self, num_blocks):
        try:
            # Sample only from tasks that request no more than existing blocks
            return self.tasks.query(f"n_blocks <= {num_blocks}").sample(1)
        except:
            logger.error(
                f"There are no tasks in the workload requesting less than {num_blocks} blocks to sample from. \
                    This workload requires at least {self.tasks['n_blocks'].min()} initial blocks"
            )
            exit(1)

    def get_task_arrival_interval_time(self):
        return random.expovariate(self.avg_num_tasks_per_block)


class CSVTaskGenerator(TaskGenerator):
    def __init__(self, df_tasks, config) -> None:
        super().__init__(df_tasks, config)

    def sample_task_row(self):
        yield self.tasks.iterrows()

    def get_task_arrival_interval_time(self):
        yield self.tasks["relative_submit_time"].iteritems()
