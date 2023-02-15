import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
import typer
import math
from precycle.utils.utils import REPO_ROOT


app = typer.Typer()


class Task:
    def __init__(
        self,
        n_blocks,
        query_id,
        utility,
        utility_beta,
        query_type,
        query,
        start_time=None,
    ):
        self.n_blocks = n_blocks
        self.query_id = query_id
        self.utility = utility
        self.utility_beta = utility_beta
        self.query_type = query_type
        self.query = query
        self.start_time = start_time


class QueryPool:
    def __init__(self, attribute_domain_sizes, queries_path) -> None:
        self.attribute_domain_sizes = attribute_domain_sizes
        self.domain_size = math.prod(attribute_domain_sizes)
        self.queries = None
        with open(queries_path) as f:
            self.queries = json.load(f)

    def get_query(self, query_id: int):
        query_id_str = str(query_id)
        if query_id_str in self.queries:
            query_vector = self.queries[query_id_str]
        assert query_vector is not None
        return query_vector


class PrivacyWorkload:
    """
    csv-based privacy workload.
    """

    def __init__(self, blocks_metadata_path, queries):
        try:
            with open(blocks_metadata_path) as f:
                blocks_metadata = json.load(f)
        except NameError:
            logger.error("Dataset metadata must have be created first..")
            exit(1)

        self.query_pool = QueryPool(blocks_metadata["attributes_domain_sizes"], queries)

    def create_dp_task(self, task) -> dict:
        task_name = f"task-{task.query_id}-{task.n_blocks}"
        dp_task = {
            "query_id": task.query_id,
            "query_type": task.query_type,
            "n_blocks": task.n_blocks,
            "utility": task.utility,
            "utility_beta": task.utility_beta,
            "task_name": task_name,
            "query": self.query_pool.get_query(task.query_id),
        }
        if task.start_time:
            dp_task.update({"submit_time": task.start_time})
        return dp_task

    def generate(self, utility, utility_beta):
        self.blocks_num = 600  # days
        self.initial_blocks_num = 1
        self.query_types = [0]
        self.requested_blocks_num = list(range(1, self.blocks_num, 50))

        def generate_one_day_tasks(
            start_time, query_types, one_task=True, all_blocks=True
        ):
            tasks = []
            num_existing_blocks = start_time + self.initial_blocks_num

            if one_task:
                std_num_tasks = 0
            else:
                std_num_tasks = 5
            if all_blocks:
                nblocks = num_existing_blocks
            else:
                nblocks_options = [
                    n for n in self.requested_blocks_num if n <= num_existing_blocks
                ]
                nblocks = np.random.choice(nblocks_options, 1)[0]

            num_tasks = np.abs(np.random.normal(1, std_num_tasks, 1)).astype(int)
            for _ in range(num_tasks[0]):
                query_id = np.random.choice(query_types)
                tasks.append(
                    Task(nblocks, query_id, utility, utility_beta, "linear", start_time)
                )
            return tasks

        self.tasks = []
        for i in range(self.blocks_num):
            self.tasks += generate_one_day_tasks(i, self.query_types)

        dp_tasks = [self.create_dp_task(t) for t in self.tasks]
        logger.info(f"Collecting results in a dataframe...")
        self.tasks = pd.DataFrame(dp_tasks).sort_values("submit_time")
        self.tasks["relative_submit_time"] = (
            self.tasks["submit_time"] - self.tasks["submit_time"].shift(periods=1)
        ).fillna(1)

        logger.info(self.tasks.head())

    def generate_nblocks(self, n_queries, rangelist, utility, utility_beta):
        # Simply lists all the queries, the sampling will happen in the simulator
        self.tasks = []

        # # Every workload has monoblocks
        # for query_id in range(n_queries):
        #     self.tasks.append(
        #         Task(
        #             n_blocks=1,
        #             query_id=query_id,
        #             utility=utility,
        #             utility_beta=utility_beta,
        #             query_type="linear",
        #             query=self.query_pool.get_query(query_id),
        #         )
        #     )
        # for b in range(nblocks_step, nblocks_max, nblocks_step):
        for b in rangelist:
            for query_id in range(n_queries):
                self.tasks.append(
                    Task(
                        n_blocks=b,
                        query_id=query_id,
                        utility=utility,
                        utility_beta=utility_beta,
                        query_type="linear",
                        query=self.query_pool.get_query(query_id),
                    )
                )
        dp_tasks = [self.create_dp_task(t) for t in self.tasks]
        logger.info(f"Collecting results in a dataframe...")

        # TODO: weird to overwrite with a different type
        self.tasks = pd.DataFrame(dp_tasks)

        logger.info(self.tasks.head())

    def dump(self, path=Path(__file__).resolve().joinpath("covid19_workload/")):
        logger.info("Saving the privacy workload...")
        self.tasks.to_csv(path, index=False)
        logger.info(f"Saved {len(self.tasks)} tasks at {path}.")


def main(
    requests_type: str = "monoblock",
    utility: float = 0.05,
    utility_beta: float = 0.001,
    queries: str = "covid19_queries/all_2way_marginals.queries.json",
    workload_dir: str = str(Path(__file__).resolve().parent.parent),
    blocks_metadata_path: str = REPO_ROOT.joinpath(
        "data/covid19/covid19_data/blocks/metadata.json"
    ),
) -> None:

    workload_dir = Path(workload_dir)
    queries = workload_dir.joinpath(queries)
    privacy_workload = PrivacyWorkload(blocks_metadata_path, queries)

    # TODO: keeping this for now to generate a very specific workload
    if requests_type == "all-blocks":
        privacy_workload.generate(utility, utility_beta)
        path = workload_dir.joinpath(f"covid19_workload/privacy_tasks.csv")

    elif requests_type == "monoblock":
        n_different_queries = len(json.load(open(queries, "r")))
        privacy_workload.generate_nblocks(
            n_different_queries,
            rangelist=[1],
            utility=utility,
            utility_beta=utility_beta,
        )
        path = workload_dir.joinpath(
            f"covid19_workload/{requests_type}.privacy_tasks.csv"
        )

    else:  # 1:1:1:2:4:8:16:32  # 3/8 to select 1 block

        rangelist = list(requests_type.split(":"))
        n_different_queries = len(json.load(open(queries, "r")))
        privacy_workload.generate_nblocks(
            n_different_queries, rangelist, utility, utility_beta
        )
        path = workload_dir.joinpath(
            f"covid19_workload/{requests_type}blocks_{n_different_queries}queries.privacy_tasks.csv"
        )
    privacy_workload.dump(path=path)


if __name__ == "__main__":
    typer.run(main)
