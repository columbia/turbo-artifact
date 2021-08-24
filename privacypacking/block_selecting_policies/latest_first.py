class LatestFirst:
    @staticmethod
    def select_blocks(blocks, task_blocks_num):
        # passing the blocks because future policies might be more elaborate and require blocks info
        blocks_num = len(blocks)
        return reversed(range(blocks_num - task_blocks_num, blocks_num))
