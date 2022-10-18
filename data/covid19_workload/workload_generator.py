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

    def __init__(self,):
        self.tasks = None
        self.blocks_num = 400
        self.num_types_of_queries = 2
        self.tasks = []
        for i in range(self.blocks_num):
            self.tasks += self.generate_one_day_tasks(i, self.num_types_of_queries)


    def generate_one_day_tasks(self, start_time, num_types_of_queries):
        tasks = []
        num_tasks = np.abs(np.random.normal(40, 20, 1)).astype(int)+1
        for _ in range(num_tasks[0]):
            query_id = np.random.randint(1, num_types_of_queries+1)-1
            query_type = "count"
            # start time is in block units, so it indicates how many blocks currently exist
            # we use this info so that a task does not request more blocks than those existing
            num_existing_blocks = start_time+1
            nblocks_options = [n for n in [1, 7, 14, 30, 60, 90, 120] if n <= num_existing_blocks]
            nblocks = np.random.choice(nblocks_options, 1)[0]
            tasks.append(Task(start_time, nblocks, query_id, query_type))

        return tasks

    def create_dp_task(self, task) -> dict:
        submit_time = task.start_time
        n_blocks = task.n_blocks
        epsilon, delta = self.compute_budget(task.query_id, n_blocks)
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
        self.tasks = pd.DataFrame(dp_tasks).sort_values('submit_time')
        self.tasks["relative_submit_time"] = (
                self.tasks["submit_time"] - self.tasks["submit_time"].shift(periods=1)
        ).fillna(0)

        logger.info(self.tasks.head())

    def dump(
        self,
        path=Path(__file__)
        .resolve()
        .parent.parent.joinpath("covid19_workload/privacy_tasks.csv"),
    ):
        # self.tasks = self.tasks.sort_values(["submit_time"])
        logger.info("Saving the privacy workload...")
        self.tasks.to_csv(path, index=False)
        logger.info(f"Saved {len(self.tasks)} tasks at {path}.")

    # Todo: this is obsolete -> users will not define their epsilon demand from now on
    def compute_budget(self, query_id, n_blocks):
        delta = 0.00001
        epsilon = [0.5]
        return epsilon, delta

    def compute_block_selection_policy(self):
        return "LatestBlocksFirst"


# @hydra.main(config_path="config", config_name="_default")
# def main(cfg: DictConfig) -> None:
def main() -> None:
    privacy_workload = PrivacyWorkload()
    privacy_workload.generate()
    privacy_workload.dump()


if __name__ == "__main__":
    main()

