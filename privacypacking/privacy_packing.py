import argparse
import pprint as pp

import yaml

from privacypacking.config import Config
from privacypacking.offline.offline_simulator import OfflineSimulator
from privacypacking.online.online_simulator import OnlineSimulator
from privacypacking.utils.utils import *

DEFAULT_CONFIG_FILE = "privacypacking/config/default_config.yaml"


class PrivacyPacking:

    def __init__(self, config_file, default_config_file):
        with open(default_config_file, 'r') as default_config:
            self.config = yaml.safe_load(default_config)
        with open(config_file, 'r') as config:
            self.user_config = yaml.safe_load(config)

        # Update the config file with the user-config's preferences
        update_dict(self.user_config, self.config)
        pp.pprint(self.config)

        config = Config(self.config)

        self.simulator = None

        # todo enable mixed block/task offline/online states in the future
        if config.blocks_offline and config.tasks_offline:
            self.config[BLOCKS_SPEC] = self.config[BLOCKS_SPEC][OFFLINE]
            self.config[TASKS_SPEC] = self.config[TASKS_SPEC][OFFLINE]
            self.simulator = OfflineSimulator(config)

        elif config.blocks_online or config.tasks_online:
            self.config[BLOCKS_SPEC] = self.config[BLOCKS_SPEC][ONLINE]
            self.config[TASKS_SPEC] = self.config[TASKS_SPEC][ONLINE]
            self.simulator = OnlineSimulator(config)

    def simulate(self):
        self.simulator.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_file')
    args = parser.parse_args()

    default_config_file = DEFAULT_CONFIG_FILE
    pp = PrivacyPacking(args.config_file, default_config_file)
    pp.simulate()
