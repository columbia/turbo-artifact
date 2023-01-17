import redis
from pricycle.budget import (
    RenyiBudget,
)

# class BudgetAccountantEntry:
#     def __init__(self, budget):
#         self.budget = budget

class BudgetAccountantKey:
    def __init__(self, block):
        self.key = f"{block}"


class BudgetAccountant:
    def __init__(self, config) -> None:
        self.config = config
        self.kv_store = redis.Redis(host=config.localhost, port=config.port, db=0)
        self.blocks_count = 0     # TODO: Initialize from KV store
        self.epsilon = float(self.config.epsilon)
        self.delta = float(self.config.delta)
        self.alphas = self.config.alphas

    def update_block_budget(self, block, budget):
        key = BudgetAccountantKey(block).key
        # Add budget in the key value store
        for alpha in budget.alphas():
            self.kv_store.hset(key, str(alpha), str(budget.epsilon(alpha)))

    def add_new_block_budget(self):
        block = self.blocks_count
        # Initialize block's budget from epsilon and delta
        budget = RenyiBudget.from_epsilon_delta(
                epsilon=self.epsilon, delta=self.delta, alpha_list=self.alphas
            )
        self.update_block_budget(block, budget)
        self.blocks_count += 1

    def get_block_budget(self, block):
        ''' Returns the remaining block budget'''
        key = BudgetAccountantKey(block).key
        orders = self.kv_store.hgetall(key)
        # TODO: convert strings to floats
        budget = RenyiBudget.from_epsilon_list(orders)
        return budget

    def can_run(self, blocks, run_budget):
        for block in blocks:
            budget = self.get_block_budget(block)
            if not budget.can_allocate(run_budget):
                return False
        return True

    def consume_block_budget(self, block, run_budget):
        ''' Consumes 'run_budget' from the remaining block budget'''
        budget = self.get_block_budget(block)
        budget -= run_budget
        # Re-write the budget in the KV store
        self.update_block_budget(block, budget)

    def consume_blocks_budget(self, blocks, run_budget):
        for block in blocks:
            self.consume_block_budget(block, run_budget)
