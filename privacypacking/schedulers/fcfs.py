from privacypacking.schedulers.scheduler import Scheduler

# TODO: see if we can reuse the greedy heuristic here
# (FCFS is a greedy heuristic with no heuristic)


class FCFS(Scheduler):
    """
    Schedule by prioritizing the tasks that come first
    """

    def __init__(self, tasks, blocks, config=None):
        super().__init__(tasks, blocks)

    def schedule(self):
        allocated_task_ids = []

        # Read them by order
        for i, task in enumerate(self.tasks):
            # self.task_set_block_ids(task)
            if self.can_run(task):
                self.consume_budgets(task)
                allocated_task_ids.append(task.id)

        return allocated_task_ids
