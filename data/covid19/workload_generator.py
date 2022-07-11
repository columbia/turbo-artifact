from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import random


def get_split(len_x, num_cuts):
    c = random.sample(list(range(0, len_x)), num_cuts)
    c.append(len_x)
    c.insert(0, 0)
    return c


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
        # self.query_instances_num = 2000
        self.blocks_num = 400
        self.num_queries = 25
        self.tasks = []
        # for i in range(self.query_instances_num):
        #     sample_query = random.randint(1, self.num_queries)
        #     self.tasks += Query(sample_query, self.blocks_num).generate_tasks()
        # print(len(self.tasks))

        # self.time_partition = sorted(get_split(self.blocks_num, 350))
        # self.freq = {}
        # for i in self.time_partition:
        #     self.freq[i] = random.randint(0, 2)
        for i in range(self.blocks_num):
            self.tasks += self.generate_one_day_tasks(i, self.num_queries)


    def generate_one_day_tasks(self, start_time, num_queries):
        tasks = []
        freq = 0
        # for i in range(len(self.time_partition)-1):
        #     if self.time_partition[i] <= start_time < self.time_partition[i + 1]:
        #         freq = self.freq[self.time_partition[i]]
        # if freq:
        freq = random.randint(0, 2)
        if freq == 0:
            num_tasks = np.abs(np.random.normal(50, 5, 1)).astype(int)+1
        # elif freq == 2:
        #     num_tasks = np.abs(np.random.normal(10, 2, 1)).astype(int)+1
        # elif freq == 1:
        #     num_tasks = np.abs(np.random.normal(20, 3, 1)).astype(int)+1
        # elif freq == 3:
        #     num_tasks = np.abs(np.random.normal(20, 2, 1)).astype(int)+1
        elif freq == 1 or freq == 2:
            num_tasks = np.abs(np.random.normal(10, 2, 1)).astype(int)+1
        # elif freq == 3:
        #     num_tasks = np.abs(np.random.normal(5, 2, 1)).astype(int)+1
        # num_tasks = np.abs(np.random.normal(20, 10, 1)).astype(int)+1
        # print(num_tasks)
        for i in range(num_tasks[0]):
            # nblocks = np.abs(np.random.normal(1, 7, 1)).astype(int)[0]+1
            # if 1 <= nblocks < 5:
            #     nblocks = 1
            # elif 5 <= nblocks < 10:
            #     nblocks = 7
            # elif 10 <= nblocks < 15:
            #     nblocks = 10
            # elif 15 <= nblocks:
            #     nblocks = 15
            query_id = np.random.randint(1, num_queries+1)
            query_type = "average"
            nblocks = np.random.choice([1, 7, 14], 1, p=[0.41, 0.39, 0.2])[0]
            # nblocks = np.random.choice([1, 7], 1, p=[0.5, 0.5])[0]
            # nblocks = np.random.choice([1, 7], 1, p=[0.5, 0.5])[0]
            # nblocks = (np.abs(np.random.normal(7, 4, 1)).astype(int)+1)[0]
            # nblocks = np.random.randint(1, 10)
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
            "profit": n_blocks,
            "block_selection_policy": self.compute_block_selection_policy(),
            "task_name": task_name,
            "alphas": [0.0],  # Hacky
            "rdp_epsilons": epsilon,
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
        .parent.parent.joinpath("covid19/privacy_tasks.csv"),
    ):
        # self.tasks = self.tasks.sort_values(["submit_time"])
        logger.info("Saving the privacy workload...")
        self.tasks.to_csv(path, index=False)
        logger.info(f"Saved {len(self.tasks)} tasks at {path}.")

    def compute_budget(self, query_id, n_blocks):
        epsilons = [0.5, 0.5]
        # epsilon = np.random.choice(epsilons, 1, p=[1/3, 1/3, 1/3])
        epsilon = np.random.choice(epsilons, 1, p=[1/2, 1/2])
        delta = 0.00001
        # epsilon = [0.5]
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

