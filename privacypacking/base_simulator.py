from privacypacking.utils.utils import *
from privacypacking.plot import Plotter

class BaseSimulator:

    def __init__(self, config):
        self.blocks_spec = config[BLOCKS_SPEC]
        self.tasks_spec = config[TASKS_SPEC]
        self.renyi_epsilon = config[RENYI_EPSILON]
        self.renyi_delta = config[RENYI_DELTA]
        self.scheduler = config[SCHEDULER]
        self.plotter = Plotter(config[PLOT_FILE])
        self.curve_distributions = self.tasks_spec[CURVE_DISTRIBUTIONS]

    def run(self):
        pass

    def plot(self, allocation, tasks, blocks):
        self.plotter.plot(tasks, blocks, allocation)
