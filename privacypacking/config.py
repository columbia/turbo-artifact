from privacypacking.logger import Logger
from privacypacking.utils.utils import *


class Config:
    def __init__(self, config):
        self.config = config
        self.renyi_epsilon = config[RENYI_EPSILON]
        self.renyi_delta = config[RENYI_DELTA]
        self.log_file = config[LOG_FILE]

        # Offline Mode
        if config[OFFLINE][ENABLED]:
            self.mode = OFFLINE
            config = config[OFFLINE]

            # Scheduler
            self.scheduler = config[SCHEDULER_SPEC]
            self.scheduler_name = self.scheduler[NAME]

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

            # Log file
            self.logger = Logger(f"privacypacking/offline/plots/{self.log_file}.log")

        # Online Mode
        elif config[ONLINE][ENABLED]:
            self.mode = ONLINE
            config = config[ONLINE]

            # Scheduler
            self.scheduler = config[SCHEDULER_SPEC]
            self.scheduler_name = self.scheduler[NAME]
            self.scheduler_N = self.scheduler[N]

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

            # Task arrival interval
            self.task_arrival_frequency = self.tasks_spec[TASK_ARRIVAL_FREQUENCY]
            if self.task_arrival_frequency[POISSON][ENABLED]:
                self.task_arrival_poisson_enabled = True
                self.poisson_random_seed = self.task_arrival_frequency[POISSON][RANDOM_SEED]
                self.task_arrival_interval = self.task_arrival_frequency[POISSON][TASK_ARRIVAL_INTERVAL]

            elif self.task_arrival_frequency[CONSTANT][ENABLED]:
                self.task_arrival_constant_enabled = True
                self.task_arrival_interval = self.task_arrival_frequency[CONSTANT][TASK_ARRIVAL_INTERVAL]
            assert self.task_arrival_poisson_enabled != self.task_arrival_constant_enabled

            # Task's block request
            self.blocks_request = self.tasks_spec[BLOCKS_REQUEST]
            if self.blocks_request[RANDOM][ENABLED]:
                self.blocks_request_random_enabled = True
                self.blocks_request_random_seed = self.blocks_request[RANDOM][RANDOM_SEED]
                self.blocks_request_random_max_num = self.blocks_request[RANDOM][BLOCKS_NUM_MAX]

            elif self.blocks_request[CONSTANT][ENABLED]:
                self.blocks_request_constant_enabled = True
                self.blocks_request_constant_num = self.blocks_request[CONSTANT][BLOCKS_NUM]
            assert self.blocks_request_random_enabled != self.blocks_request_constant_enabled

            self.block_selecting_policy = self.tasks_spec[BLOCK_SELECTING_POLICY]

            # Log file
            self.logger = Logger(f"privacypacking/online/plots/{self.log_file}.log")

        self.laplace_noise_start = self.laplace[NOISE_START]
        self.laplace_noise_stop = self.laplace[NOISE_STOP]
        self.gaussian_sigma_start = self.gaussian[SIGMA_START]
        self.gaussian_sigma_stop = self.gaussian[SIGMA_STOP]
        self.subsamplegaussian_sigma_start = self.subsamplegaussian[SIGMA_START]
        self.subsamplegaussian_sigma_stop = self.subsamplegaussian[SIGMA_STOP]
        self.subsamplegaussian_dataset_size = self.subsamplegaussian[DATASET_SIZE]
        self.subsamplegaussian_batch_size = self.subsamplegaussian[BATCH_SIZE]
        self.subsamplegaussian_epochs = self.subsamplegaussian[EPOCHS]
