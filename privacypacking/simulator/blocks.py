from itertools import count
from loguru import logger
from privacypacking.budget import Block


class Blocks:
    """
    Model block arrival.
    """

    def __init__(self, environment, resource_manager):
        self.env = environment
        self.resource_manager = resource_manager
        self.omegaconf = self.resource_manager.omegaconf
        self.blocks_count = count()

        if self.omegaconf.scheduler.method == "offline":
            self.omegaconf.blocks.max_num = self.omegaconf.blocks.initial_num

        self.env.process(self.block_producer())

    def block_producer(self):
        """Generate blocks."""
        # Produce initial blocks
        for _ in range(self.omegaconf.blocks.initial_num):
            self.env.process(self.block(next(self.blocks_count)))

        for _ in range(
            self.omegaconf.blocks.max_num - self.omegaconf.blocks.initial_num
        ):
            block_id = next(self.blocks_count)
            self.env.process(self.block(block_id))
            yield self.env.timeout(self.omegaconf.blocks.arrival_interval)

        self.resource_manager.block_production_terminated.succeed()

    def block(self, block_id):
        """
        Block behavior. Sets its own budget, notifies resource manager of its existence,
        waits till it gets generated
        """
        block = self.create_block(block_id)
        generated_block_event = self.env.event()
        yield self.resource_manager.new_blocks_queue.put((block, generated_block_event))
        yield generated_block_event
        logger.debug(f"Block: {block_id} generated at {self.env.now}")

    def create_block(self, block_id: int) -> Block:
        # TODO: add a flag to switch between pure/renyi dp
        # block = Block(
        #     block_id,
        #     BasicBudget(self.omegaconf.epsilon),
        # )
        block = Block.from_epsilon_delta(
            block_id,
            self.omegaconf.epsilon,
            self.omegaconf.delta,
            alpha_list=self.omegaconf.alphas,
        )
        return block
