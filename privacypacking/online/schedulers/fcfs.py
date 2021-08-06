from privacypacking.online.schedulers.scheduler import Scheduler

# TODO: see if we can reuse the greedy heuristic here
# (FCFS is a greedy heuristic with no heuristic)


class FCFS(Scheduler):
    """
    Schedule by prioritizing the tasks that come first
    """

    def __init__(self, tasks, blocks, config=None):
        super().__init__(tasks, blocks, config)

    def schedule(self):
        allocation = [False] * len(self.tasks)

        # todo: lock block
        # Read them by order
        for i, task in enumerate(self.tasks):
            if self.can_run(task):
                self.consume_budgets(task)
                allocation[i] = True
        return allocation
