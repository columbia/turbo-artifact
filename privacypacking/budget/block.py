from privacypacking.budget import Budget


class Block:
    def __init__(self, id, budget):
        self.id = id
        self.budget = budget
        # add other properties here

def create_block(block_id, e, d):
    # Same budget per block for now
    block = Block(block_id, Budget.from_epsilon_delta(epsilon=e, delta=d))
    return block

