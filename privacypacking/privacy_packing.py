import typer
from omegaconf import OmegaConf

from privacypacking.config import Config
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import DEFAULT_CONFIG_FILE, save_logs


def main(cfg: str = typer.Option("")):
    if cfg:
        omegaconf = OmegaConf.load(cfg)
        conf = Config({"omegaconf": omegaconf})
    else:
        conf = Config({})
    logs = Simulator(conf).run()
    save_logs(conf, logs)


if __name__ == "__main__":
    typer.run(main)
