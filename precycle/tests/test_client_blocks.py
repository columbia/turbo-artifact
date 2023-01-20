import typer
from omegaconf import OmegaConf
from precycle.client_blocks import BlocksClient
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

    block_data_path = config.blocks.block_data_path + "/block_1.csv"
    BlocksClient(config.blocks_server).send_request(block_data_path)


if __name__ == "__main__":
    app()
