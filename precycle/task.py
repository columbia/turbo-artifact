from typing import Union, List


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
        name: str = None,
    ):
        self.id = id
        self.query_id = query_id
        self.query_type = query_type
        self.query = query
        self.blocks = blocks
        self.n_blocks = n_blocks
        self.utility = utility
        self.utility_beta = utility_beta
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
