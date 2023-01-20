import json
import typer
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from precycle.utils.utils import DEFAULT_CONFIG_FILE
from precycle.budget import SparseHistogram

app = typer.Typer()


class MockPSQLConnection:
    def __init__(self, config) -> None:
        self.config = config
        # Blocks are in-memory histograms
        self.blocks = {}
        self.blocks_count = 0

        try:
            with open(config.blocks.blocks_metadata_path) as f:
                self.blocks_metadata = json.load(f)
        except NameError:
            logger.error("Dataset metadata must have be created first..")
            exit(1)

        self.attributes_domain_sizes = self.blocks_metadata["attributes_domain_sizes"]


    def add_new_block(self, block_data_path):
        raw_data = pd.read_csv(block_data_path)
        histogram_data = SparseHistogram.from_dataframe(
            raw_data, self.attributes_domain_sizes
        )
        self.blocks[self.blocks_count] = histogram_data
        self.blocks_count += 1

    def run_query(self, query, blocks):
        result = 0
        for block in range(blocks[0], blocks[1]+1):
            result += self.blocks[block].run(query)
        return result
        

    def close(self):
        pass


@app.command()
def run(
    omegaconf: str = "precycle/config/precycle.json",
):
    omegaconf = OmegaConf.load(omegaconf)
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconf = OmegaConf.create(omegaconf)
    config = OmegaConf.merge(default_config, omegaconf)

if __name__ == "__main__":
    app()
