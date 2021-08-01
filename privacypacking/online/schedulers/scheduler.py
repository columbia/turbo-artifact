from privacypacking.plot import plot


class Scheduler:
    def __init__(self, tasks, blocks):
        self.tasks = tasks
        self.blocks = blocks

    def schedule(self):
        pass

    def plot(self, allocation):
        plot(self.tasks, self.blocks, allocation)
