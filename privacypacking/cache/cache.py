class R:
    def __init__(self, query_id, blocks, budget) -> None:
        self.query_id = query_id
        self.blocks = blocks
        self.budget = budget

    def __str__(
        self,
    ):
        return f"Run({self.blocks})"


class A:
    def __init__(self, l, budget=None) -> None:
        self.l = l
        self.budget = budget

    def __str__(
        self,
    ):
        return f"Aggregate({[str(l) for l in self.l]})"


class Cache:
    def __init__(
        self,
    ):
        pass

    def dump(
        self,
    ):
        pass

    def run(
        self,
    ):
        pass

    def get_execution_plan(
        self,
    ):
        pass