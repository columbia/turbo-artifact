import random
from loguru import logger
from privacypacking.budget import Task
from privacypacking.budget.block_selection import (
    BlockSelectionPolicy,
    LatestBlocksFirst,
)


class TaskGenerator:
    def __init__(self, df_tasks) -> None:
        self.tasks = df_tasks

    def sample_task_row(self):
        raise NotImplementedError("Must override")

    def create_task(self, task_id):
        task_row = self.sample_task_row().squeeze()

        profit = 1 if "profit" not in task_row else float(task_row["profit"])
        name = task_id if "task_name" not in task_row else task_row["task_name"]

        task = Task(
            id=task_id,
            query_id=int(task_row["query_id"]),
            query_type=task_row["query_type"],
            profit=profit,
            block_selection_policy=BlockSelectionPolicy.from_str(
                str(task_row["block_selection_policy"])
            ),
            n_blocks=int(task_row["n_blocks"]),
            utility=float(task_row["utility"]),
            utility_beta=float(task_row["utility_beta"]),
            name=name,
        )
        return task


class PoissonTaskGenerator(TaskGenerator):
    def __init__(self, df_tasks, avg_num_tasks_per_block, system_blocks) -> None:
        super().__init__(df_tasks)
        self.blocks = system_blocks
        self.avg_num_tasks_per_block = avg_num_tasks_per_block

    def sample_task_row(self):
        nblocks = len(self.blocks)
        try:
            # Sample only from tasks that request no more than existing blocks
            return self.tasks.query(f"n_blocks <= {nblocks}").sample(1)
        except:
            logger.error(
                f"There are no tasks in the workload requesting less than {nblocks} blocks to sample from. \
                    This workload requires at least {self.tasks['n_blocks'].min()} initial blocks"
            )
            exit(1)

    def get_task_arrival_interval_time(self):
        return random.expovariate(self.avg_num_tasks_per_block)


class CSVTaskGenerator(TaskGenerator):
    def __init__(self, df_tasks) -> None:
        super().__init__(df_tasks)

    def sample_task_row(self):
        yield self.tasks.iterrows()

    def get_task_arrival_interval_time(self):
        yield self.tasks["relative_submit_time"].iteritems()
