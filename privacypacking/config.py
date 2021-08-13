from datetime import datetime

from privacypacking.logger import Logger
from privacypacking.utils.utils import *


# TODO: SimulatorConfig? Also, refactor the CLI or PrivacyPacking class
class Config:
    def __init__(self, config):
        self.config = config
        self.global_seed = config[GLOBAL_SEED]
        self.deterministic = config[DETERMINISTIC]
        self.epsilon = config[EPSILON]
        self.delta = config[DELTA]

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
            self.block_arrival_frequency = self.blocks_spec[BLOCK_ARRIVAL_FRQUENCY]
            if self.block_arrival_frequency[ENABLED]:
                if self.block_arrival_frequency[POISSON][ENABLED]:
                    self.block_arrival_poisson_enabled = True
                    self.block_arrival_constant_enabled = False
                    self.block_arrival_interval = self.block_arrival_frequency[POISSON][
                        BLOCK_ARRIVAL_INTERVAL
                    ]
                if self.block_arrival_frequency[CONSTANT][ENABLED]:
                    self.block_arrival_constant_enabled = True
                    self.block_arrival_poisson_enabled = False
                    self.block_arrival_interval = self.block_arrival_frequency[
                        CONSTANT
                    ][BLOCK_ARRIVAL_INTERVAL]
            else:
                self.block_arrival_interval = None

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

            # Task's block request
            self.blocks_request = self.tasks_spec[BLOCKS_REQUEST]
            if self.blocks_request[RANDOM][ENABLED]:
                self.blocks_request_random_enabled = True
                self.blocks_request_constant_enabled = False
                self.blocks_request_random_max_num = self.blocks_request[RANDOM][
                    BLOCKS_NUM_MAX
                ]

            elif self.blocks_request[CONSTANT][ENABLED]:
                self.blocks_request_constant_enabled = True
                self.blocks_request_random_enabled = False
                self.blocks_request_constant_num = self.blocks_request[CONSTANT][
                    BLOCKS_NUM
                ]

            self.block_selecting_policy = self.tasks_spec[BLOCK_SELECTING_POLICY]

            # Log file

        if LOG_FILE in config:
            self.log_file = f"{config[LOG_FILE]}.json"
        else:
            self.log_file = f"{self.mode}/{self.scheduler_name}/{datetime.now().strftime('%m%d-%H%M%S')}.json"
        log_path = LOGS_PATH.joinpath(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(log_path)

        self.laplace_noise_start = self.laplace[NOISE_START]
        self.laplace_noise_stop = self.laplace[NOISE_STOP]
        self.gaussian_sigma_start = self.gaussian[SIGMA_START]
        self.gaussian_sigma_stop = self.gaussian[SIGMA_STOP]
        self.subsamplegaussian_sigma_start = self.subsamplegaussian[SIGMA_START]
        self.subsamplegaussian_sigma_stop = self.subsamplegaussian[SIGMA_STOP]
        self.subsamplegaussian_dataset_size = self.subsamplegaussian[DATASET_SIZE]
        self.subsamplegaussian_batch_size = self.subsamplegaussian[BATCH_SIZE]
        self.subsamplegaussian_epochs = self.subsamplegaussian[EPOCHS]

    def dump(self) -> dict:
        return self.config
