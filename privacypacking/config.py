import math
import random
import uuid
from datetime import datetime
from functools import partial
from typing import List

from privacypacking.budget import Block, Task
from privacypacking.budget.curves import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
)
from privacypacking.budget.task import UniformTask
from privacypacking.logger import Logger
from privacypacking.schedulers.utils import (
    TASK_BASED_BUDGET_UNLOCKING,
    THRESHOLD_UPDATING,
    TIME_BASED_BUDGET_UNLOCKING,
)
from privacypacking.utils.utils import *


# Configuration Reading Logic
class Config:
    def __init__(self, config):
        self.config = config
        self.epsilon = config[EPSILON]
        self.delta = config[DELTA]

        # DETERMINISM
        self.global_seed = config[GLOBAL_SEED]
        self.deterministic = config[DETERMINISTIC]
        if self.deterministic:
            random.seed(self.global_seed)
            np.random.seed(self.global_seed)

        # SCHEDULER
        self.scheduler = config[SCHEDULER_SPEC]
        self.scheduler_method = self.scheduler[METHOD]
        self.scheduler_metric = self.scheduler[METRIC]
        self.scheduler_N = self.scheduler[N]
        self.scheduler_budget_unlocking_time = self.scheduler[BUDGET_UNLOCKING_TIME]
        self.scheduler_scheduling_wait_time = self.scheduler[SCHEDULING_WAIT_TIME]

        self.scheduler_threshold_update_mechanism = self.scheduler[
            THRESHOLD_UPDATE_MECHANISM
        ]
        self.new_task_driven_scheduling = False
        self.time_based_scheduling = False
        self.new_block_driven_scheduling = False
        if self.scheduler_method == THRESHOLD_UPDATING:
            self.new_task_driven_scheduling = True
            self.new_block_driven_scheduling = True
        elif self.scheduler_method == TIME_BASED_BUDGET_UNLOCKING:
            self.time_based_scheduling = True
        else:
            self.new_task_driven_scheduling = True

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

        # Setting config for "custom" tasks
        self.data_path = self.curve_distributions[CUSTOM][DATA_PATH]
        self.data_task_frequencies_path = self.curve_distributions[CUSTOM][
            DATA_TASK_FREQUENCIES_PATH
        ]
        self.custom_tasks_init_num = self.curve_distributions[CUSTOM][INITIAL_NUM]
        self.custom_tasks_frequency = self.curve_distributions[CUSTOM][FREQUENCY]
        self.custom_tasks_sampling = self.curve_distributions[CUSTOM][SAMPLING]

        self.custom_read_block_selection_policy_from_config = False
        if self.curve_distributions[CUSTOM][READ_BLOCK_SELECTION_POLICY_FROM_CONFIG][
            ENABLED
        ]:
            self.custom_read_block_selection_policy_from_config = True

        self.task_frequencies_file = None

        if self.data_path != "":
            self.data_path = REPO_ROOT.joinpath("data").joinpath(self.data_path)
            self.tasks_path = self.data_path.joinpath("tasks")
            self.task_frequencies_path = self.data_path.joinpath(
                "task_frequencies"
            ).joinpath(self.data_task_frequencies_path)

            with open(self.task_frequencies_path, "r") as f:
                self.task_frequencies_file = yaml.safe_load(f)
            assert len(self.task_frequencies_file) > 0

        # Setting config for laplace tasks
        self.laplace = self.curve_distributions[LAPLACE]
        self.laplace_init_num = self.laplace[INITIAL_NUM]
        self.laplace_frequency = self.laplace[FREQUENCY]
        self.laplace_noise_start = self.laplace[NOISE_START]
        self.laplace_noise_stop = self.laplace[NOISE_STOP]

        # Setting config for gaussian tasks
        self.gaussian = self.curve_distributions[GAUSSIAN]
        self.gaussian_init_num = self.gaussian[INITIAL_NUM]
        self.gaussian_frequency = self.gaussian[FREQUENCY]
        self.gaussian_sigma_start = self.gaussian[SIGMA_START]
        self.gaussian_sigma_stop = self.gaussian[SIGMA_STOP]

        # Setting config for subsampledGaussian tasks
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

        self.max_tasks = None
        if self.tasks_spec[MAX_TASKS][ENABLED]:
            self.max_tasks = self.tasks_spec[MAX_TASKS][NUM]

        # Log file
        if LOG_FILE in config:
            self.log_file = f"{config[LOG_FILE]}.json"
        else:
            self.log_file = f"{self.scheduler_method}_{self.scheduler_metric}/{datetime.now().strftime('%m%d-%H%M%S')}_{str(uuid.uuid4())[:6]}.json"

        if CUSTOM_LOG_PREFIX in config:
            self.log_path = LOGS_PATH.joinpath(config[CUSTOM_LOG_PREFIX]).joinpath(
                self.log_file
            )
        else:
            self.log_path = LOGS_PATH.joinpath(self.log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(
            self.log_path, f"{self.scheduler_method}_{self.scheduler_metric}"
        )
        self.log_every_n_iterations = config[LOG_EVERY_N_ITERATIONS]

    def dump(self) -> dict:
        return self.config

    # Utils to initialize tasks and blocks. It only depends on the configuration, not on the simulator.
    def set_curve_distribution(self) -> str:
        curve = np.random.choice(
            [CUSTOM, GAUSSIAN, LAPLACE, SUBSAMPLEGAUSSIAN],
            1,
            p=[
                self.custom_tasks_frequency,
                self.gaussian_frequency,
                self.laplace_frequency,
                self.subsamplegaussian_frequency,
            ],
        )
        return curve[0]

    def set_task_num_blocks(self, curve: str, max_num_blocks: int = math.inf) -> int:
        task_blocks_num = None
        block_requests = self.curve_distributions[curve][BLOCKS_REQUEST]
        if block_requests[RANDOM][ENABLED]:
            task_blocks_num = random.randint(1, block_requests[RANDOM][NUM_BLOCKS_MAX])
        elif block_requests[CONSTANT][ENABLED]:
            task_blocks_num = block_requests[CONSTANT][NUM_BLOCKS]
        task_blocks_num = max(1, min(task_blocks_num, max_num_blocks))
        assert task_blocks_num is not None
        return task_blocks_num

    def create_task(
        self, task_id: int, curve_distribution: str, num_blocks: int
    ) -> Task:

        task = None

        if curve_distribution is None:
            # If curve is not pre-specified (as in offline setting) then sample one
            curve_distribution = self.set_curve_distribution()

        if curve_distribution == CUSTOM:
            # Read custom task specs from a file
            if self.custom_tasks_sampling:
                files = [
                    f"{self.tasks_path}/{task_file}"
                    for task_file in self.task_frequencies_file.keys()
                ]
                frequencies = [
                    task_frequency
                    for task_frequency in self.task_frequencies_file.values()
                ]
                file = np.random.choice(
                    files,
                    1,
                    p=frequencies,
                )[0]

                task_spec = self.load_task_spec_from_file(file)
                if self.custom_read_block_selection_policy_from_config:
                    block_selection_policy = BlockSelectionPolicy.from_str(
                        self.curve_distributions[curve_distribution][
                            READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
                        ][BLOCK_SELECTING_POLICY]
                    )
                else:
                    block_selection_policy = task_spec.block_selection_policy

                assert block_selection_policy is not None

                task = UniformTask(
                    id=task_id,
                    profit=task_spec.profit,
                    block_selection_policy=block_selection_policy,
                    n_blocks=task_spec.n_blocks,
                    budget=task_spec.budget,
                )
        else:
            # Sample the specs of the task
            task_num_blocks = self.set_task_num_blocks(curve_distribution, num_blocks)
            block_selection_policy = BlockSelectionPolicy.from_str(
                self.curve_distributions[curve_distribution][BLOCK_SELECTING_POLICY]
            )

            if curve_distribution == GAUSSIAN:
                sigma = random.uniform(
                    self.gaussian_sigma_start, self.gaussian_sigma_stop
                )
                task = UniformTask(
                    id=task_id,
                    profit=self.set_profit(),
                    block_selection_policy=block_selection_policy,
                    n_blocks=task_num_blocks,
                    budget=GaussianCurve(sigma),
                )
            elif curve_distribution == LAPLACE:
                noise = random.uniform(
                    self.laplace_noise_start, self.laplace_noise_stop
                )
                task = UniformTask(
                    id=task_id,
                    profit=self.set_profit(),
                    block_selection_policy=block_selection_policy,
                    n_blocks=task_num_blocks,
                    budget=LaplaceCurve(noise),
                )
            elif curve_distribution == SUBSAMPLEGAUSSIAN:
                sigma = random.uniform(
                    self.subsamplegaussian_sigma_start,
                    self.subsamplegaussian_sigma_stop,
                )
                task = UniformTask(
                    id=task_id,
                    profit=self.set_profit(),
                    block_selection_policy=block_selection_policy,
                    n_blocks=task_num_blocks,
                    budget=SubsampledGaussianCurve.from_training_parameters(
                        self.subsamplegaussian_dataset_size,
                        self.subsamplegaussian_batch_size,
                        self.subsamplegaussian_epochs,
                        sigma,
                    ),
                )
        assert task is not None
        return task

    def create_block(self, block_id: int) -> Block:
        return Block.from_epsilon_delta(block_id, self.epsilon, self.delta)

    def set_profit(self):
        return 1

    def set_task_arrival_time(self):
        task_arrival_interval = None
        if self.task_arrival_poisson_enabled:
            task_arrival_interval = partial(
                random.expovariate, 1 / self.task_arrival_interval
            )()
        elif self.task_arrival_constant_enabled:
            task_arrival_interval = self.task_arrival_interval
        assert task_arrival_interval is not None
        return task_arrival_interval

    def set_block_arrival_time(self):
        block_arrival_interval = None
        if self.block_arrival_poisson_enabled:
            block_arrival_interval = partial(
                random.expovariate, 1 / self.block_arrival_interval
            )()
        elif self.block_arrival_constant_enabled:
            block_arrival_interval = self.block_arrival_interval
        assert block_arrival_interval is not None
        return block_arrival_interval

    def get_initial_task_curves(self) -> List[str]:
        curves = (
            [LAPLACE] * self.laplace_init_num
            + [GAUSSIAN] * self.gaussian_init_num
            + [SUBSAMPLEGAUSSIAN] * self.subsamplegaussian_init_num
            + [CUSTOM] * self.custom_tasks_init_num
        )
        random.shuffle(curves)
        return curves

    def get_initial_tasks_num(self) -> int:
        return (
            self.laplace_init_num
            + self.gaussian_init_num
            + self.subsamplegaussian_init_num
            + self.custom_tasks_init_num
        )

    def get_initial_blocks_num(self) -> int:
        return self.initial_blocks_num

    # todo: transferred here temporarily so that fixed seed applies for those random choices as well
    def load_task_spec_from_file(self, path: Path = PRIVATEKUBE_DEMANDS_PATH) -> TaskSpec:

        with open(path, "r") as f:
            demand_dict = yaml.safe_load(f)
            orders = {}
            for i, alpha in enumerate(demand_dict["alphas"]):
                orders[alpha] = demand_dict["rdp_epsilons"][i]
            block_selection_policy = None
            if "block_selection_policy" in demand_dict:
                block_selection_policy = BlockSelectionPolicy.from_str(
                    demand_dict["block_selection_policy"]
                )

            # Select num of blocks
            n_blocks_requests = demand_dict["n_blocks"].split(",")
            num_blocks = [
                n_blocks_request.split(":")[0] for n_blocks_request in n_blocks_requests
            ]
            frequencies = [
                n_blocks_request.split(":")[1] for n_blocks_request in n_blocks_requests
            ]
            n_blocks = np.random.choice(
                num_blocks,
                1,
                p=frequencies,
            )[0]

            # Select profit
            if "profit" in demand_dict:
                profit_requests = demand_dict["profit"].split(",")
                profits = [
                    profit_request.split(":")[0] for profit_request in profit_requests
                ]
                frequencies = [
                    profit_request.split(":")[1] for profit_request in profit_requests
                ]
                profit = np.random.choice(
                    profits,
                    1,
                    p=frequencies,
                )[0]
            else:
                profit = 1

            task_spec = TaskSpec(
                profit=float(profit),
                block_selection_policy=block_selection_policy,
                n_blocks=int(n_blocks),
                budget=Budget(orders),
            )
        assert task_spec is not None
        return task_spec