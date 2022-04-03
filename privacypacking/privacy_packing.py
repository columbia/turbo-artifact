from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import save_logs
from privacypacking.config import Config
from privacypacking.utils.utils import DEFAULT_CONFIG_FILE
from omegaconf import OmegaConf


def main():
    conf = Config({})
    logs = Simulator(conf).run()
    save_logs(conf, logs)


if __name__ == "__main__":
    main()
