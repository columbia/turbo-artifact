import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
import typer

app = typer.Typer()


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
        self.blocks_num = 600  # days
        self.initial_blocks_num = 1
        self.query_types = [0]  # [33479, 34408]
        # self.std_num_tasks = 5
        # self.requested_blocks_num = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
        #   ------------  /Configure  ------------ #

        self.tasks = []
        for i in range(self.blocks_num):
            self.tasks += self.generate_one_day_tasks(i, self.query_types)

    def generate_one_day_tasks(self, start_time, query_types):
        tasks = []
        num_existing_blocks = start_time + self.initial_blocks_num

        if self.one_task:
            std_num_tasks = 0
        else:
            std_num_tasks = 5
        if self.all_blocks:
            nblocks = num_existing_blocks
        else:
            nblocks_options = [
                n for n in self.requested_blocks_num if n <= num_existing_blocks
            ]
            nblocks = np.random.choice(nblocks_options, 1)[0]

        num_tasks = np.abs(np.random.normal(1, std_num_tasks, 1)).astype(int)

        for _ in range(num_tasks[0]):
            query_id = np.random.choice(query_types)
            query_type = "count"
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

        self.tasks = []
        for i in range(self.blocks_num):
            self.tasks += self.generate_one_day_tasks(i, self.query_types)

        dp_tasks = [self.create_dp_task(t) for t in self.tasks]
        logger.info(f"Collecting results in a dataframe...")
        self.tasks = pd.DataFrame(dp_tasks).sort_values("submit_time")
        self.tasks["relative_submit_time"] = (
            self.tasks["submit_time"] - self.tasks["submit_time"].shift(periods=1)
        ).fillna(1)

        logger.info(self.tasks.head())

    def generate_nblocks(self, n_queries, n_blocks=1):
        # Simply lists all the queries, the sampling will happen in the simulator

        self.tasks = []

        for b in range(1, n_blocks + 1):
            for query_id in range(n_queries):
                self.tasks.append(
                    Task(
                        start_time=0, n_blocks=b, query_id=query_id, query_type="linear"
                    )
                )

        dp_tasks = [self.create_dp_task(t) for t in self.tasks]
        logger.info(f"Collecting results in a dataframe...")

        # TODO: weird to overwrite with a different type
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
    def compute_budget(
        self,
    ):
        delta = 100  #      # those user defined values don't matter now so I won't them
        epsilon = [1000]  # [0.5]     # to cause trouble if code takes them into account
        return epsilon, delta

    def compute_block_selection_policy(self):
        return "LatestBlocksFirst"


def main(
    requests_type: str = "monoblock",
    queries: str = "covid19_queries/all_2way_marginals.queries.json",
    workload_dir: str = str(Path(__file__).resolve().parent.parent),
) -> None:
    privacy_workload = PrivacyWorkload()

    # TODO: refactor more at some point, especially if we add more workloads.
    # workload -> arrival and requests pattern, covid19 -> covid19-workload, separate generation scripts from the workload data

    workload_dir = Path(workload_dir)
    queries = workload_dir.joinpath(queries)

    if requests_type == "" and queries == "":
        privacy_workload.generate()
        path = workload_dir.joinpath(f"covid19_workload/privacy_tasks.csv")
    elif requests_type == "monoblock":
        n_different_queries = len(json.load(open(queries, "r")))
        privacy_workload.generate_nblocks(n_different_queries, n_blocks=1)
        path = workload_dir.joinpath(
            f"covid19_workload/{requests_type}.privacy_tasks.csv"
        )
    else:
        n_blocks = int(requests_type)
        n_different_queries = len(json.load(open(queries, "r")))
        privacy_workload.generate_nblocks(n_different_queries, n_blocks=n_blocks)
        path = workload_dir.joinpath(
            f"covid19_workload/{n_blocks}blocks_{n_different_queries}queries.privacy_tasks.csv"
        )

    privacy_workload.dump(path=path)


if __name__ == "__main__":
    typer.run(main)
