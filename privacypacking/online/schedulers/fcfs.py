from privacypacking.online.schedulers.scheduler import Scheduler


class FCFS(Scheduler):
    def __init__(self, tasks, blocks):
        super().__init__(tasks, blocks)

    def schedule(self):
        # todo

        # Schedule all tasks for now
        task_idxs = [i for i, _ in enumerate(self.tasks)]

        return task_idxs
