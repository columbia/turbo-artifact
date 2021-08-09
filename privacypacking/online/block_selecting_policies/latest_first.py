from privacypacking.online.block_selecting_policies.block_selecting_policy import BlockSelectingPolicy


class LatestFirst(BlockSelectingPolicy):
    def __init__(self, blocks, task_blocks_num):
        super().__init__(blocks, task_blocks_num)

    def select_blocks(self):
        blocks_num = len(self.blocks)
        return reversed(range(blocks_num - self.task_blocks_num, blocks_num))
