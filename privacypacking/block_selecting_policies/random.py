import random

random.seed(44)


class Random:
    @staticmethod
    def select_blocks(blocks, task_blocks_num):
        # BUG: fails if task_blocks_num > len(blocks)
        blocks_num = range(len(blocks))
        return random.sample(blocks_num, task_blocks_num)
