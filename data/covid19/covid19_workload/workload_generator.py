from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger


class Task:
    def __init__(self, start_time, n_blocks, query_id, query_type):
        self.start_time = start_time
        self.n_blocks = n_blocks
        self.query_id = query_id
        self.query_type = query_type


class PrivacyWorkload:
    """
    csv-based privacy workload.
    """

    def __init__(
        self,
    ):
        self.tasks = None

        #   ------------  Configure  ------------ #
        self.blocks_num = 100   # days
        self.initial_blocks_num = 1
        self.query_types = [0] #[33479, 34408]
        self.std_num_tasks = 5
        # self.requested_blocks_num = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800] 
        #   ------------  /Configure  ------------ #

        self.tasks = []
        for i in range(self.blocks_num):
            self.tasks += self.generate_one_day_tasks(i, self.query_types)

    def generate_one_day_tasks(self, start_time, query_types):
        tasks = []
        # num_tasks = (
        #     np.abs(np.random.normal(1, self.std_num_tasks, 1)).astype(int) + 1
        # )
        num_tasks = [1]
        for _ in range(num_tasks[0]):
            query_id = np.random.choice(query_types)
            query_type = "count"
            # start time is in block units, so it indicates how many blocks currently exist
            # we use this info so that a task does not request more blocks than those existing
            num_existing_blocks = start_time+self.initial_blocks_num
            # nblocks_options = [
            #     n for n in self.requested_blocks_num if n <= num_existing_blocks
            # ]
            # nblocks = np.random.choice(nblocks_options, 1)[0]
            nblocks = num_existing_blocks
            tasks.append(Task(start_time, nblocks, query_id, query_type))
        return tasks

    def create_dp_task(self, task) -> dict:
        submit_time = task.start_time
        n_blocks = task.n_blocks
        epsilon, delta = self.compute_budget()
        task_name = f"task-{task.query_id}-{n_blocks}-{submit_time}"

        task = {
            "query_id": task.query_id,
            "query_type": task.query_type,
            "epsilon": epsilon[0],
            "delta": delta,
            "n_blocks": n_blocks,
            "profit": 1,
            "block_selection_policy": self.compute_block_selection_policy(),
            "task_name": task_name,
            "submit_time": submit_time,
        }
        return task

    def generate(self):
        dp_tasks = [self.create_dp_task(t) for t in self.tasks]
        logger.info(f"Collecting results in a dataframe...")
        self.tasks = pd.DataFrame(dp_tasks).sort_values("submit_time")
        self.tasks["relative_submit_time"] = (
            self.tasks["submit_time"] - self.tasks["submit_time"].shift(periods=1)
        ).fillna(1)

        logger.info(self.tasks.head())

    def dump(
        self,
        path=Path(__file__)
        .resolve()
        .parent.parent.joinpath("covid19_workload/privacy_tasks.csv"),
    ):
        logger.info("Saving the privacy workload...")
        self.tasks.to_csv(path, index=False)
        logger.info(f"Saved {len(self.tasks)} tasks at {path}.")

    # Todo: this is obsolete -> users will not define their epsilon demand from now on
    def compute_budget(self,):
        delta = 0.00001
        epsilon = [0.001]
        return epsilon, delta

    def compute_block_selection_policy(self):
        return "LatestBlocksFirst"


def main() -> None:
    privacy_workload = PrivacyWorkload()
    privacy_workload.generate()
    privacy_workload.dump()


if __name__ == "__main__":
    main()
