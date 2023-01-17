from typing import Any, Iterable, Union, List
from precycle.budget.renyi_budget import ALPHAS
from precycle.utils.utils import sample_one_from_string


class Task:
    def __init__(
        self,
        id: int,
        query_id: int,
        query_type: str,
        blocks: List[int],
        n_blocks: Union[int, str],
        utility: float,
        utility_beta: float,
        name: str = None,
    ):
        self.id = id
        self.query_id = query_id
        self.query_type = query_type
        self.blocks = blocks
        self.n_blocks = n_blocks
        self.utility = utility
        self.utility_beta = utility_beta
        self.name = name

    def sample_n_blocks_and_profit(self):
        """
        If profit and n_blocks are stochastic, we sample their value when the task is added to the scheduler.
        Do not cache this for all the instances of a same task, unless this is intended.
        """

        if isinstance(self.n_blocks, str):
            self.n_blocks = int(sample_one_from_string(self.n_blocks))

        if isinstance(self.profit, str):
            self.profit = sample_one_from_string(self.profit)

    def dump(self):
        d = {
            "id": self.id,
            "query_id": self.query_id,
            "blocks": self.blocks0,
            "start_time": None,
            "allocation_time": None,
            "n_blocks": len(self.blocks),
            "max_block_id": max(self.blocks),
        }
        return d
