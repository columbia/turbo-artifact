import argparse
import yaml
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import update_dict, save_logs, DEFAULT_CONFIG_FILE
from privacypacking.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_file")
    args = parser.parse_args()

    with open(DEFAULT_CONFIG_FILE, "r") as default_config:
        config = yaml.safe_load(default_config)
    with open(args.config_file, "r") as user_config:
        user_config = yaml.safe_load(user_config)

    # Update the config file with the user-config's preferences
    update_dict(user_config, config)
    # pp.pprint(self.config)
    conf = Config(config)
    logs = Simulator(conf).run()
    save_logs(conf, logs)


if __name__ == "__main__":
    main()
