import random

random.seed(44)


class Random:
    @staticmethod
    def select_blocks(blocks, task_blocks_num):
        blocks_num = range(len(blocks))
        return random.sample(blocks_num, task_blocks_num)
