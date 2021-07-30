import random

from privacypacking.budget.block import *
from privacypacking.budget.task import *
from privacypacking.offline.schedulers.scheduler import Scheduler


class PierreHeuristic(Scheduler):
    def __init__(self, tasks, blocks):
        super().__init__(tasks, blocks)

    def schedule(self):
        allocation = []
        return allocation


def main():
    # num_blocks = 1 # single-block case
    num_blocks = 2  # multi-block case

    blocks = [create_block(i, 10, 0.001) for i in range(num_blocks)]
    tasks = ([create_gaussian_task(i, num_blocks, range(num_blocks), s) for i, s in
              enumerate(np.linspace(0.1, 1, 10))] +
             [create_gaussian_task(i, num_blocks, range(num_blocks), l) for i, l in
              enumerate(np.linspace(0.1, 10, 5))] +
             [create_subsamplegaussian_task(i, num_blocks, range(num_blocks),
                                            ds=60_000,
                                            bs=64,
                                            epochs=10,
                                            s=s) for i, s in enumerate(np.linspace(1, 10, 5))]
             )

    random.shuffle(tasks)
    scheduler = PierreHeuristic(tasks, blocks)
    allocation = scheduler.schedule()
    scheduler.plot(allocation)


if __name__ == "__main__":
    main()
