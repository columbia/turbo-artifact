import argparse
import os
from datetime import datetime
from distutils.log import set_verbosity
from typing import Any, Dict, List

import loguru
import yaml
from loguru import logger
from ray import tune
from ray.tune import Stopper
from ray.tune.stopper import TimeoutStopper

from privacypacking.config import Config
from privacypacking.schedulers.utils import (
    ARGMAX_KNAPSACK,
    BASIC_SCHEDULER,
    BATCH_OVERFLOW_RELEVANCE,
    DOMINANT_SHARES,
    DYNAMIC_FLAT_RELEVANCE,
    FCFS,
    FLAT_RELEVANCE,
    NAIVE_AVERAGE,
    OVERFLOW_RELEVANCE,
    SIMPLEX,
    SOFT_KNAPSACK,
    TIME_BASED_BUDGET_UNLOCKING,
)
from privacypacking.simulator.simulator import Simulator
from privacypacking.utils.utils import *


class TrialStopper(Stopper):
    def __init__(self, max_seconds=10):
        self._deadline = max_seconds

    def __call__(self, trial_id, result):
        logger.warning(
            f"Trial {trial_id} has been running for {result['time_total_s']} seconds"
        )
        return result["time_total_s"] > self._deadline

    def stop_all(self):
        return False


def run_and_report(config: dict) -> None:
    # Unpack conditional parameters
    config[SCHEDULER_SPEC][METHOD], config[SCHEDULER_SPEC][METRIC] = config.pop(
        "method_and_metric"
    )

    logger.info(f"Running simulator with config: {config}")

    sim = Simulator(Config(config))
    metrics = sim.run()
    # logger.info(metrics)

    logger.info(f"Trial logs: {tune.get_trial_dir()}")

    tune.report(**metrics)


def grid():

    with open(DEFAULT_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    with open(
        DEFAULT_CONFIG_FILE.parent.joinpath(
            "offline_dpf_killer/multi_block/gap_base.yaml"
        ),
        "r",
    ) as user_config:
        user_config = yaml.safe_load(user_config)
    update_dict(user_config, config)

    temperature = [1]

    # temperature = [1, 10, 1e4]
    # temperature = [2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 5e-2]
    temperature = [
        # 1e9,
        # 1e8,
        # 1e7,
        # 1e6,
        # 5e5,
        # 1e5,
        # 5e4,
        # 1e4,
        # 5e3,
        # 4e3,
        # 3e3,
        # 2e3,
        1750,
        1500,
        1100,
        1010,
        1001,
        1e3,
        # 1e2,
        # 1e-1,
        # 1,
        # 1e-1,
        # 1e-2,
        # 1e-3,
        # 1e-4,
        # 1e-5,
    ]

    normalize_by = ["available_budget"]
    metric_recomputation_period = [1_000]

    # monoalpha = [3, 4, 5, 8, 16, 64]

    # Conditonal parameter
    method_and_metric = []
    for metric in [
        # DOMINANT_SHARES,
        # FLAT_RELEVANCE,
        # OVERFLOW_RELEVANCE,
        # FCFS,
        SOFT_KNAPSACK,
        # ARGMAX_KNAPSACK,
        # DYNAMIC_FLAT_RELEVANCE,
    ]:
        method_and_metric.append((BASIC_SCHEDULER, metric))
    # method_and_metric.append((SIMPLEX, DOMINANT_SHARES))

    config["method_and_metric"] = tune.grid_search(method_and_metric)

    # num_tasks = [50, 100, 200, 300, 350, 400, 500, 750, 1000, 1500, 2000]
    num_tasks = [20_000]
    # num_tasks = [100]

    num_blocks = [20]
    data_path = "privatekube_event_g0.0_l0.5_p=grid"
    block_selection_policies = ["RandomBlocks"]

    config[BLOCKS_SPEC][INITIAL_NUM] = tune.grid_search(num_blocks)
    config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM].update(
        {
            SAMPLING: True,
            INITIAL_NUM: tune.grid_search(num_tasks),
            DATA_PATH: data_path,
            DATA_TASK_FREQUENCIES_PATH: "frequencies.yaml",
            FREQUENCY: 1,
            READ_BLOCK_SELECTION_POLICY_FROM_CONFIG: {
                ENABLED: True,
                BLOCK_SELECTING_POLICY: tune.grid_search(block_selection_policies),
            },
        }
    )
    # config[SCHEDULER_SPEC][METHOD] = TIME_BASED_BUDGET_UNLOCKING

    # config[BLOCKS_SPEC][INITIAL_NUM] = tune.grid_search(num_blocks)
    # # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][SAMPLING] = True
    # # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][INITIAL_NUM] = tune.grid_search(
    # #     num_tasks
    # # )
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][INITIAL_NUM] = tune.grid_search(
    #     num_tasks
    # )
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
    #     READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    # ][ENABLED] = True
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
    #     READ_BLOCK_SELECTION_POLICY_FROM_CONFIG
    # ][BLOCK_SELECTING_POLICY] = tune.grid_search(block_selection_policies)
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][DATA_PATH] = data_path
    # config[TASKS_SPEC][CURVE_DISTRIBUTIONS][CUSTOM][
    #     DATA_TASK_FREQUENCIES_PATH
    # ] = "frequencies.yaml"

    config[CUSTOM_LOG_PREFIX] = f"exp_{datetime.now().strftime('%m%d-%H%M%S')}"

    config["omegaconf"] = {
        "scheduler": {
            "metric_recomputation_period": tune.grid_search(
                metric_recomputation_period
            ),
            "log_warning_every_n_allocated_tasks": 1_000,
            "scheduler_timeout_seconds": 20 * 60,
        },
        "metric": {
            "normalize_by": tune.grid_search(normalize_by),
            "temperature": tune.grid_search(temperature),
            "gurobi_timeout": 1_000,
            "gurobi_threads": 1,
            "n_knapsack_solvers": 16,
            "save_profit_matrix": True,
        },
        "logs": {
            "verbose": False,
            "save": True,
        },
        # "alphas": [tune.grid_search(monoalpha)],
    }

    tune.run(
        run_and_report,
        config=config,
        resources_per_trial={"cpu": 3},
        local_dir=RAY_LOGS,
        resume=False,
        # progress_reporter=CustomReporter(),
        verbose=0,
        callbacks=[
            CustomLoggerCallback(),
            tune.logger.JsonLoggerCallback(),
            tune.integration.mlflow.MLflowLoggerCallback(
                experiment_name="fair_tasks",
            ),
        ],
        # stop=TrialStopper(max_seconds=30), # the stopper isn't triggered unless it returns a result
        # stop=TimeoutStopper(timeout=10), # the runner doesn't listen to the interupt call
    )


class CustomLoggerCallback(tune.logger.LoggerCallback):
    """Custom logger interface"""

    def __init__(self) -> None:
        super().__init__()

    def log_trial_result(self, iteration: int, trial: Any, result: Dict):
        logger.info(
            [
                f"{key}: {result[key]}"
                for key in ["n_allocated_tasks", "realized_profit", "temperature"]
            ]
        )
        return

    def on_trial_complete(self, iteration: int, trials: List, trial: Any, **info):
        return


# def __init__(self, filename: str = "log.txt):
#     self._trial_files = {}
#     self._filename = filename

# def log_trial_start(self, trial: "Trial"):
#     trial_logfile = os.path.join(trial.logdir, self._filename)
#     self._trial_files[trial] = open(trial_logfile, "at")

# def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
#     if trial in self._trial_files:
#         self._trial_files[trial].write(json.dumps(result))

# def on_trial_complete(self, iteration: int, trials: List["Trial"],
#                       trial: "Trial", **info):
#     if trial in self._trial_files:
#         self._trial_files[trial].close()
#         del self._trial_files[trial]


class CustomReporter(tune.CLIReporter):
    def __init__(self):
        super().__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len(
            [t for t in trials if t.status == tune.trial.Trial.TERMINATED]
        )
        return self.num_terminated > old_num_terminated

    def log_result(self, trial: "Trial", result: Dict, error: bool = False):
        return

    # def report(self, trials, *sys_info):
    #     print("done")
    #     # print(*sys_info)
    #     # print("\n".join([str(trial) for trial in trials]))


if __name__ == "__main__":
    # os.environ["LOGURU_LEVEL"] = "INFO"
    os.environ["LOGURU_LEVEL"] = "WARNING"
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    grid()
