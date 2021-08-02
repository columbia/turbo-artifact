from typing import List


class Scheduler:
    def __init__(self, tasks, blocks):
        self.tasks = tasks
        self.blocks = blocks

    def schedule(self) -> List[bool]:
        pass
