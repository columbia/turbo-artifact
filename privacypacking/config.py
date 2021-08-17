from datetime import datetime

from privacypacking.logger import Logger
from privacypacking.utils.utils import *


class Config:
    def __init__(self, config):
        self.config = config
        self.global_seed = config[GLOBAL_SEED]
        self.deterministic = config[DETERMINISTIC]
        self.epsilon = config[EPSILON]
        self.delta = config[DELTA]

        # SCHEDULER
        self.scheduler = config[SCHEDULER_SPEC]
        self.scheduler_name = self.scheduler[NAME]
        self.scheduler_N = self.scheduler[N]

        # BLOCKS
        self.blocks_spec = config[BLOCKS_SPEC]
        self.initial_blocks_num = self.blocks_spec[INITIAL_NUM]
        self.block_arrival_frequency = self.blocks_spec[BLOCK_ARRIVAL_FRQUENCY]
        if self.block_arrival_frequency[ENABLED]:
            self.block_arrival_frequency_enabled = True
            if self.block_arrival_frequency[POISSON][ENABLED]:
                self.block_arrival_poisson_enabled = True
                self.block_arrival_constant_enabled = False
                self.block_arrival_interval = self.block_arrival_frequency[POISSON][
                    BLOCK_ARRIVAL_INTERVAL
                ]
            if self.block_arrival_frequency[CONSTANT][ENABLED]:
                self.block_arrival_constant_enabled = True
                self.block_arrival_poisson_enabled = False
                self.block_arrival_interval = self.block_arrival_frequency[CONSTANT][
                    BLOCK_ARRIVAL_INTERVAL
                ]
        else:
            self.block_arrival_frequency_enabled = False

        # TASKS
        self.tasks_spec = config[TASKS_SPEC]
        self.curve_distributions = self.tasks_spec[CURVE_DISTRIBUTIONS]

        # Setting config for laplace tasks
        self.laplace = self.curve_distributions[LAPLACE]
        self.laplace_init_num = self.laplace[INITIAL_NUM]
        self.laplace_frequency = self.laplace[FREQUENCY]
        self.laplace_noise_start = self.laplace[NOISE_START]
        self.laplace_noise_stop = self.laplace[NOISE_STOP]

        self.gaussian = self.curve_distributions[GAUSSIAN]
        self.gaussian_init_num = self.gaussian[INITIAL_NUM]
        self.gaussian_frequency = self.gaussian[FREQUENCY]
        self.gaussian_sigma_start = self.gaussian[SIGMA_START]
        self.gaussian_sigma_stop = self.gaussian[SIGMA_STOP]

        self.subsamplegaussian = self.curve_distributions[SUBSAMPLEGAUSSIAN]
        self.subsamplegaussian_init_num = self.subsamplegaussian[INITIAL_NUM]
        self.subsamplegaussian_frequency = self.subsamplegaussian[FREQUENCY]
        self.subsamplegaussian_sigma_start = self.subsamplegaussian[SIGMA_START]
        self.subsamplegaussian_sigma_stop = self.subsamplegaussian[SIGMA_STOP]
        self.subsamplegaussian_dataset_size = self.subsamplegaussian[DATASET_SIZE]
        self.subsamplegaussian_batch_size = self.subsamplegaussian[BATCH_SIZE]
        self.subsamplegaussian_epochs = self.subsamplegaussian[EPOCHS]

        self.task_arrival_frequency = self.tasks_spec[TASK_ARRIVAL_FREQUENCY]
        if self.task_arrival_frequency[ENABLED]:
            self.task_arrival_frequency_enabled = True

            if self.task_arrival_frequency[POISSON][ENABLED]:
                self.task_arrival_poisson_enabled = True
                self.task_arrival_constant_enabled = False
                self.task_arrival_interval = self.task_arrival_frequency[POISSON][
                    TASK_ARRIVAL_INTERVAL
                ]

            elif self.task_arrival_frequency[CONSTANT][ENABLED]:
                self.task_arrival_constant_enabled = True
                self.task_arrival_poisson_enabled = False
                self.task_arrival_interval = self.task_arrival_frequency[CONSTANT][
                    TASK_ARRIVAL_INTERVAL
                ]
            assert (
                self.task_arrival_poisson_enabled != self.task_arrival_constant_enabled
            )
        else:
            self.task_arrival_frequency_enabled = False

        # Log file
        if LOG_FILE in config:
            self.log_file = f"{config[LOG_FILE]}.json"
        else:
            self.log_file = (
                f"{self.scheduler_name}/{datetime.now().strftime('%m%d-%H%M%S')}.json"
            )
        log_path = LOGS_PATH.joinpath(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(log_path, self.scheduler_name)

    def dump(self) -> dict:
        return self.config
