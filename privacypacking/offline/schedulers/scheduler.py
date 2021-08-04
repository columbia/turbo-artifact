from typing import List

# QUESTION: should `schedule` modify the blocks inplace? (reduce the budget available)
# TODO: should we store a dict of blocks, to avoid `get_block_by_id`?


# Or should it just return an allocation, that we execute in a second step?
class Scheduler:
    def __init__(self, tasks, blocks):
        self.tasks = tasks
        self.blocks = blocks

    def schedule(self) -> List[bool]:
        pass
