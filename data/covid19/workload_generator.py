import math
import os
from pathlib import Path
from typing import Optional, Union
# import hydra
import pandas as pd
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from queries import Query, generate_one_day_tasks
import random


class PrivacyWorkload:
    """
    csv-based privacy workload.
    """

    def __init__(self,):
        self.tasks = None
        # self.query_instances_num = 2000
        self.blocks_num = 400
        self.num_queries = 5
        self.tasks = []
        # for i in range(self.query_instances_num):
        #     sample_query = random.randint(1, self.num_queries)
        #     self.tasks += Query(sample_query, self.blocks_num).generate_tasks()
        # print(len(self.tasks))
        for i in range(self.blocks_num):
            self.tasks += generate_one_day_tasks(i, self.num_queries)

    def create_dp_task(self, task) -> dict:
        submit_time = task.start_time
        n_blocks = task.n_blocks
        epsilon, delta = self.compute_budget(task.type, n_blocks)
        task_name = f"task-{task.type}-{n_blocks}-{submit_time}"

        task = {
            "query_type": task.type,
            "epsilon": epsilon[0],
            "delta": delta,
            "n_blocks": n_blocks,
            "profit": 1,
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

    def compute_budget(self, task_type, n_blocks):
        epsilons = [0.5]
        # epsilon = np.random.choice(epsilons, 1, p=[1/2, 1/2])
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

