import pprint as pp

import yaml

from privacypacking.offline.offline_simulator import OfflineSimulator
from privacypacking.online.online_simulator import OnlineSimulator
from privacypacking.utils.utils import *
import argparse

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

        blocks_spec = self.config[BLOCKS_SPEC]
        tasks_spec = self.config[TASKS_SPEC]
        self.simulator = None

        # todo enable mixed block/task offline/online states in the future
        if blocks_spec[OFFLINE][ENABLED] and tasks_spec[OFFLINE][ENABLED]:
            self.config[BLOCKS_SPEC] = self.config[BLOCKS_SPEC][OFFLINE]
            self.config[TASKS_SPEC] = self.config[TASKS_SPEC][OFFLINE]
            self.simulator = OfflineSimulator(self.config)

        elif blocks_spec[ONLINE][ENABLED] or tasks_spec[ONLINE][ENABLED]:
            self.config[BLOCKS_SPEC] = self.config[BLOCKS_SPEC][ONLINE]
            self.config[TASKS_SPEC] = self.config[TASKS_SPEC][ONLINE]
            self.simulator = OnlineSimulator(self.config)

    def simulate(self):
        self.simulator.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_file')
    args = parser.parse_args()

    default_config_file = DEFAULT_CONFIG_FILE
    pp = PrivacyPacking(args.config_file, default_config_file)
    pp.simulate()
