import typer
from omegaconf import OmegaConf
from precycle.budget import RenyiBudget
from precycle.utils.utils import DEFAULT_CONFIG_FILE

app = typer.Typer()


class BudgetAccountantKey:
    def __init__(self, block):
        self.key = f"{block}"


class MockBudgetAccountant:
    def __init__(self, config) -> None:
        self.config = config
        # key-value store is just an in-memory dictionary
        self.kv_store = {}
        self.epsilon = float(self.config.epsilon)
        self.delta = float(self.config.delta)
        self.alphas = self.config.alphas

    def get_blocks_count(self):
        return len(self.kv_store.keys())

    def update_block_budget(self, block, budget):
        key = BudgetAccountantKey(block).key
        # Add budget in the key value store
        self.kv_store[key] = budget

    def add_new_block_budget(self):
        block = self.get_blocks_count()
        # Initialize block's budget from epsilon and delta
        budget = RenyiBudget.from_epsilon_delta(
            epsilon=self.epsilon, delta=self.delta, alpha_list=self.alphas
        )
        self.update_block_budget(block, budget)

    def get_block_budget(self, block):
        """Returns the remaining block budget"""
        key = BudgetAccountantKey(block).key
        if key in self.kv_store:
            budget = self.kv_store[key]
            return budget
        return None

    def can_run(self, blocks, run_budget):
        for block in range(blocks[0], blocks[1]+1):
            budget = self.get_block_budget(block)
            print(budget)
            if not budget.can_allocate(run_budget):
                return False
        return True

    def consume_block_budget(self, block, run_budget):
        """Consumes 'run_budget' from the remaining block budget"""
        budget = self.get_block_budget(block)
        budget -= run_budget
        # Re-write the budget in the KV store
        self.update_block_budget(block, budget)

    def consume_blocks_budget(self, blocks, run_budget):
        for block in blocks:
            self.consume_block_budget(block, run_budget)


@app.command()
def run(
    omegaconf: str = "precycle/config/precycle.json",
):
    omegaconf = OmegaConf.load(omegaconf)
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconf = OmegaConf.create(omegaconf)
    config = OmegaConf.merge(default_config, omegaconf)

    budget_accountant = MockBudgetAccountant(config=config.budget_accountant)
    budget_accountant.add_new_block_budget()


if __name__ == "__main__":
    app()
