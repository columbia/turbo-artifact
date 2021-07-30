from privacypacking.utils.utils import *


class BaseSimulator:

    def __init__(self, config):
        self.blocks_spec = config[BLOCKS_SPEC]
        self.tasks_spec = config[TASKS_SPEC]
        self.renyi_epsilon = config[RENYI_EPSILON]
        self.renyi_delta = config[RENYI_DELTA]
        self.scheduler = config[SCHEDULER]

    def run(self):
        pass
