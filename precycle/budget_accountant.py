import redis
from loguru import logger
from precycle.budget import RenyiBudget, BasicBudget


class BudgetAccountantKey:
    def __init__(self, block):
        self.key = f"{block}"


class BudgetAccountant:
    def __init__(self, config) -> None:
        self.config = config.budget_accountant
        self.kv_store = redis.Redis(host=config.host, port=config.port, db=0)
        self.epsilon = float(self.config.epsilon)
        self.delta = float(self.config.delta)
        self.alphas = self.config.alphas

    def get_blocks_count(self):
        return len(self.kv_store.keys("*"))

    def update_block_budget(self, block, budget):
        key = BudgetAccountantKey(block).key
        # Add budget in the key value store
        for alpha in budget.alphas:
            self.kv_store.hset(key, str(alpha), str(budget.epsilon(alpha)))

    def add_new_block_budget(self):
        block = self.get_blocks_count()
        # Initialize block's budget from epsilon and delta
        budget = RenyiBudget.from_epsilon_delta(epsilon=self.epsilon, delta=self.delta)
        self.update_block_budget(block, budget)

    def get_block_budget(self, block):
        """Returns the remaining block budget"""
        key = BudgetAccountantKey(block).key
        orders = self.kv_store.hgetall(key)
        alphas = [float(alpha) for alpha in orders.keys()]
        epsilons = [float(epsilon) for epsilon in orders.values()]
        budget = RenyiBudget.from_epsilon_list(epsilons, alphas)
        return budget

    def get_all_block_budgets(self):
        raise NotImplementedError

    def can_run(self, blocks, run_budget):
        for block in range(blocks[0], blocks[1] + 1):
            budget = self.get_block_budget(block)
            if not budget.can_allocate(run_budget):
                return False
        return True

    def consume_block_budget(self, block, run_budget):
        """Consumes 'run_budget' from the remaining block budget"""
        budget = self.get_block_budget(block)
        budget -= run_budget
        # Re-write the budget in the KV store
        self.update_block_budget(block, budget)


class MockBudgetAccountant:
    def __init__(self, config) -> None:
        self.config = config
        # key-value store is just an in-memory dictionary
        self.kv_store = {}
        self.epsilon = float(self.config.budget_accountant.epsilon)
        self.delta = float(self.config.budget_accountant.delta)
        self.alphas = self.config.budget_accountant.alphas

    def get_blocks_count(self):
        return len(self.kv_store.keys())

    def update_block_budget(self, block, budget):
        key = BudgetAccountantKey(block).key
        # Add budget in the key value store
        self.kv_store[key] = budget

    def add_new_block_budget(self):
        block = self.get_blocks_count()
        # Initialize block's budget from epsilon and delta
        budget = (
            BasicBudget(self.epsilon)
            if self.config.puredp
            else RenyiBudget.from_epsilon_delta(epsilon=self.epsilon, delta=self.delta)
        )
        self.update_block_budget(block, budget)

    def get_block_budget(self, block):
        """Returns the remaining budget of block"""
        key = BudgetAccountantKey(block).key
        if key in self.kv_store:
            budget = self.kv_store[key]
            return budget
        # logger.info(f"Block {block} does not exist")
        return None

    def get_all_block_budgets(self):
        return self.kv_store.items()

    def can_run(self, blocks, run_budget):
        for block in range(blocks[0], blocks[1] + 1):
            budget = self.get_block_budget(block)
            if not budget.can_allocate(run_budget):
                return False
        return True

    def consume_block_budget(self, block, run_budget):
        """Consumes 'run_budget' from the remaining block budget"""
        budget = self.get_block_budget(block)
        budget -= run_budget
        # Re-write the budget in the KV store
        self.update_block_budget(block, budget)

    def dump(self):
        budgets = [
            (block, budget.dump()) for (block, budget) in self.get_all_block_budgets()
        ]
        return budgets
