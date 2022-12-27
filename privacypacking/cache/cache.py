from typing import Dict, Tuple

from privacypacking.budget import Budget


class R:
    def __init__(self, blocks, noise_std) -> None:
        self.blocks = blocks
        self.noise_std = noise_std

    def __str__(self,):
        return f"Run({self.blocks}, {self.noise_std})"


class A:
    def __init__(self, query_id, l) -> None:
        self.l = l
        self.query_id = query_id

    def __str__(self,):
        return f"Aggregate({[str(l) for l in self.l]})"


class Cache:
    def __init__(self):
        pass

    def dump(self):
        pass

    def run(self) -> Tuple[float, Budget, Dict]:
        pass

    def get_execution_plan(self):
        pass
