import pickle
import random
import time

import numpy as np
from loguru import logger

from precycle.cache.histogram import k_way_marginal_query_list
from precycle.query_converter import QueryConverter
from precycle.task import Task


def Zipf(a: np.float64, min: np.uint64, max: np.uint64, size=None):
    """
    Generate Zipf-like random variables,
    but in inclusive [min...max] interval
    """
    if min == 0:
        raise ZeroDivisionError("")

    v = np.arange(min, max + 1)  # values to sample
    p = 1.0 / np.power(v, a)  # probabilities
    p /= np.sum(p)
    return np.random.choice(v, size=size, replace=True, p=p)


class TaskGenerator:
    def __init__(self, df_tasks, config) -> None:
        self.config = config
        self.tasks = df_tasks
        self.query_converter = QueryConverter(self.config.blocks_metadata)
        self.query_pool = {}

    def sample_task_row(self, config):
        raise NotImplementedError("Must override")

    def create_task(self, task_id, num_blocks):
        task_row = self.sample_task_row(num_blocks, self.config).squeeze()

        query_id = int(task_row["query_id"])
        name = task_id if "task_name" not in task_row else task_row["task_name"]
        # We sample only tasks whose n_blocks is <= num_blocks
        num_requested_blocks = int(task_row["n_blocks"])

        if self.config.tasks.block_selection_policy == "LatestBlocks":
            requested_blocks = (num_blocks - num_requested_blocks, num_blocks - 1)
        elif self.config.tasks.block_selection_policy == "RandomBlocks":
            start = np.random.randint(0, num_blocks - num_requested_blocks + 1)
            requested_blocks = (start, start + num_requested_blocks - 1)

        # t = time.time()
        # Query may be either a query_vector (covid19) or a dict format (citibike)
        query = eval(task_row["query"])

        # print(f"Read query from csv: {query} of type {type(query)}")

        # Read compressed rectangle or PyTorch slice, output a query vector
        if isinstance(query, dict):
            # NOTE: we only support pure k-way marginals for now
            attribute_sizes = self.config.blocks_metadata["attributes_domain_sizes"]
            query_vector = k_way_marginal_query_list(
                query, attribute_sizes=attribute_sizes
            )
        else:
            query_vector = query

        if query_id in self.query_pool:
            query_tensor = self.query_pool[query_id]
        else:
            # Load tensor /query from disk if stored
            if "query_path" in task_row:
                with open(task_row["query_path"], "rb") as f:
                    query_tensor = pickle.load(f)
            else:
                query_tensor = self.query_converter.convert_to_sparse_tensor(query)

            # Converting to dense tensor to facilitate future tensor operations
            # TODO: Maybe this is unecessary? I'm not very familiar with PyTorch.
            query_tensor = query_tensor.to_dense()
            # self.query_pool[query_id] = query_tensor

        query_db_format = (
            query_tensor
            if self.config.mock
            else self.query_converter.convert_to_sql(query_vector, task.blocks)
        )
        # print("Query Prep Time", time.time() - t)

        task = Task(
            id=task_id,
            query_id=query_id,
            query_type=task_row["query_type"],
            query=query_tensor,
            query_db_format=query_db_format,
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
        self.tasks = self.tasks.sample(
            frac=1, random_state=config.global_seed
        ).reset_index()

        def zipf():
            query_pool_size = len(self.tasks["query_id"].unique())
            min = np.uint64(1)
            max = np.uint64(query_pool_size)
            samples = Zipf(config.tasks.zipf_k, min, max, int(config.tasks.max_num))
            for sample in samples:
                yield sample
            # return samples

        self.zipf_sampling = zipf()
        # print("\n\n\nUnique queries", len([*set(self.zipf_sampling)]))
        # exit()

    def sample_task_row(self, num_blocks, config):
        try:
            next_sample_idx = int(next(self.zipf_sampling)) - 1
            next_query_id = self.tasks.iloc[[next_sample_idx]]["query_id"].values[0]
            # next_query_id = next_sample_idx
            # Sample only from tasks that request no more than existing blocks
            return self.tasks.query(
                f"n_blocks <= {num_blocks} and query_id == {next_query_id}"
            ).sample(1, random_state=int(random.random() * 100))

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

    def sample_task_row(self, config):
        yield self.tasks.iterrows()

    def get_task_arrival_interval_time(self):
        yield self.tasks["relative_submit_time"].iteritems()
