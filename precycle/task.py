from typing import List, Union

from precycle.budget.histogram import k_way_marginal_query_list


class Task:
    def __init__(
        self,
        id: int,
        query_id: int,
        query_type: str,
        query: List[List[int]],
        blocks: List[int],
        n_blocks: Union[int, str],
        utility: float,
        utility_beta: float,
        query_db_format=None,
        name: str = None,
        attribute_sizes: List[int] = None,
    ):
        self.id = id
        self.query_id = query_id
        self.query_type = query_type

        # Read compressed rectangle or PyTorch slice, output a query vector
        if isinstance(query, dict):
            # NOTE: we only support pure k-way marginals for now
            self.query = k_way_marginal_query_list(
                query, attribute_sizes=attribute_sizes
            )
        else:
            self.query = query
        self.blocks = blocks
        self.n_blocks = n_blocks
        self.utility = utility
        self.utility_beta = utility_beta
        self.query_db_format = query_db_format
        self.name = name

    def dump(self):
        d = {
            "id": self.id,
            "query_id": self.query_id,
            # "query": self.query,
            "blocks": self.blocks,
            "n_blocks": self.n_blocks,
            "utility": self.utility,
            "utility_beta": self.utility_beta,
            "name": self.name,
        }
        return d


class TaskInfo:
    def __init__(self, task, status, planning_time, run_metadta, result) -> None:
        self.d = task.dump()
        self.d.update(
            {
                "status": status,
                "planning_time": planning_time,
                "run_metadata": run_metadta,
                "result": result,
            }
        )

    def dump(self):
        return self.d
