"""Script to launch sequences of experiments.
Runs in parallel with Ray and gathers the hyperparameters and results in a TensorBoard.

Usage: modify this script with the configuration logic you need.
"""

import yaml
from ray import tune
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from privacypacking.privacy_packing_simulator import run
from privacypacking.utils.utils import *


def run_and_report(config: dict) -> None:

    metrics = run(config)

    tune.report(**metrics)


def main():

    with open(DEFAULT_CONFIG_FILE, "r") as f:
        default_config = yaml.safe_load(f)

    # A list of configuration parameters that override the default parameters
    search_space = []

    for scheduler_name in ["OfflineDPF", "FlatRelevance", "OverflowRelevance"]:
        extra_config = {"scheduler_spec": {"name": scheduler_name}}
        search_space.append(extra_config)

    # (I had to apply this patch: https://github.com/ray-project/ray/pull/17282/files)
    # Comment out the warning in lib/python3.9/site-packages/ray/tune/suggest/variant_generator.py"
    analysis = tune.run(
        run_and_report,
        config=default_config,
        resources_per_trial={"cpu": 1},
        local_dir=RAY_LOGS,
        resume=False,
        search_alg=BasicVariantGenerator(points_to_evaluate=search_space),
        num_samples=len(search_space),
    )


if __name__ == "__main__":
    main()
