import typer
from omegaconf import OmegaConf
from precycle.budget_accounant import BudgetAccountant
from precycle.utils.utils import DEFAULT_CONFIG_FILE

app = typer.Typer()

@app.command()
def run(
    omegaconf: str = "precycle/config/precycle.json",
):
    omegaconf = OmegaConf.load(omegaconf)
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconf = OmegaConf.create(omegaconf)
    config = OmegaConf.merge(default_config, omegaconf)

    budget_accountant = BudgetAccountant(config=config.budget_accountant)
    budget_accountant.add_new_block_budget()


if __name__ == "__main__":
    app()
