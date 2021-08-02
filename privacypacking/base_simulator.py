from privacypacking.utils.utils import *
from privacypacking.plot import Plotter

class BaseSimulator:

    def __init__(self, config):
       self.config = config

    def run(self):
        pass

    def plot(self, allocation, tasks, blocks):
        self.config.plotter.plot(tasks, blocks, allocation)
