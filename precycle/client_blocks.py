import json
import typer
import socket
from omegaconf import OmegaConf
from precycle.utils.utils import DEFAULT_CONFIG_FILE

app = typer.Typer()


class BlocksClient:
    def __init__(self, config) -> None:
        self.config = config
        self.host = self.config.host
        self.port = self.config.port

    def send_request(self, block_data_path):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            s.sendall(block_data_path)
            data = s.recv(1024)
        print(f"Received {data!r}")



@app.command()
def run(
    omegaconf: str = "precycle/config/precycle.json",
):
    omegaconf = OmegaConf.load(omegaconf)
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconf = OmegaConf.create(omegaconf)
    config = OmegaConf.merge(default_config, omegaconf)
    
    block_data_path = config.blocks_server.block_data_path + "/block_1.csv"
    data = bytes(block_data_path, 'utf-8')

    BlocksClient(config.blocks_server).send_request(data)


if __name__ == "__main__":
    app()
