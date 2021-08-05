from privacypacking.plot import Plotter
from privacypacking.utils.utils import *


class Config:

    def __init__(self, config):
        self.config = config
        self.renyi_epsilon = config[RENYI_EPSILON]
        self.renyi_delta = config[RENYI_DELTA]
        self.plotter = Plotter(config[PLOT_FILE])

        # Offline Mode
        if config[OFFLINE][ENABLED]:
            self.mode = OFFLINE
            config = config[OFFLINE]

            # Blocks
            self.blocks_spec = config[BLOCKS_SPEC]
            self.blocks_num = self.blocks_spec[NUM]

            # Tasks
            self.tasks_spec = config[TASKS_SPEC]
            self.curve_distributions = self.tasks_spec[CURVE_DISTRIBUTIONS]
            self.laplace = self.curve_distributions[LAPLACE]
            self.gaussian = self.curve_distributions[GAUSSIAN]
            self.subsamplegaussian = self.curve_distributions[SUBSAMPLEGAUSSIAN]
            self.laplace_num = self.laplace[NUM]
            self.subsamplegaussian_num = self.subsamplegaussian[NUM]
            self.gaussian_num = self.gaussian[NUM]


        # Online Mode
        elif config[ONLINE][ENABLED]:
            self.mode = ONLINE
            config = config[ONLINE]

            # Blocks
            self.blocks_spec = config[BLOCKS_SPEC]
            self.blocks_num = self.blocks_spec[NUM]
            self.block_arrival_interval = self.blocks_spec[BLOCK_ARRIVAL_INTERVAL]

            # Tasks
            self.tasks_spec = config[TASKS_SPEC]
            self.curve_distributions = self.tasks_spec[CURVE_DISTRIBUTIONS]
            self.laplace = self.curve_distributions[LAPLACE]
            self.gaussian = self.curve_distributions[GAUSSIAN]
            self.subsamplegaussian = self.curve_distributions[SUBSAMPLEGAUSSIAN]
            self.laplace_frequency = self.laplace[FREQUENCY]
            self.gaussian_frequency = self.gaussian[FREQUENCY]
            self.subsamplegaussian_frequency = self.subsamplegaussian[FREQUENCY]
            self.task_arrival_interval = self.tasks_spec[TASK_ARRIVAL_INTERVAL]

        self.laplace_noise_start = self.laplace[NOISE_START]
        self.laplace_noise_stop = self.laplace[NOISE_STOP]
        self.gaussian_sigma_start = self.gaussian[SIGMA_START]
        self.gaussian_sigma_stop = self.gaussian[SIGMA_STOP]
        self.subsamplegaussian_sigma_start = self.subsamplegaussian[SIGMA_START]
        self.subsamplegaussian_sigma_stop = self.subsamplegaussian[SIGMA_STOP]
        self.subsamplegaussian_dataset_size = self.subsamplegaussian[DATASET_SIZE]
        self.subsamplegaussian_batch_size = self.subsamplegaussian[BATCH_SIZE]
        self.subsamplegaussian_epochs = self.subsamplegaussian[EPOCHS]

        # Scheduler
        self.scheduler = config[SCHEDULER]
