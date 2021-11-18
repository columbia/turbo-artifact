from pathlib import Path

import numpy as np
import typer
import yaml
from loguru import logger

from privacypacking.budget import ALPHAS
from privacypacking.budget.curves import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
)
from privacypacking.budget.utils import compute_noise_from_target_epsilon

DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent.parent.joinpath("data/mixed_curves/")


def main(
    epsilon_1: float = typer.Option(0.5, help="Low budget tasks"),
    epsilon_2: float = typer.Option(1.0, help="High budget tasks"),
    delta: float = typer.Option(1e-5, help="Delta"),
    n_1: int = typer.Option(1, help="Low number of blocks"),
    n_2: int = typer.Option(10, help="High number of blocks"),
    block_selection_policy: str = typer.Option(
        "RandomBlocks", help="Block selection policy"
    ),
    output_path: str = typer.Option(str(DEFAULT_OUTPUT_PATH)),
):

    output_path = Path(output_path)

    tasks_path = output_path.joinpath("tasks")
    tasks_path.mkdir(exist_ok=True, parents=True)

    frequencies_path = output_path.joinpath("task_frequencies")
    frequencies_path.mkdir(exist_ok=True, parents=True)

    task_names = []
    for curve_type in ["laplace", "gaussian", "subsampled_gaussian"]:
        for epsilon in [epsilon_1, epsilon_2]:

            if curve_type == "subsampled_gaussian":

                # Typical values for MNIST-like datasets
                dataset_size = 50_000
                batch_size = 100
                epochs = 50

                sigma = compute_noise_from_target_epsilon(
                    target_epsilon=epsilon,
                    target_delta=delta,
                    epochs=epochs,
                    batch_size=batch_size,
                    dataset_size=dataset_size,
                )
                budget = SubsampledGaussianCurve.from_training_parameters(
                    dataset_size=dataset_size,
                    batch_size=batch_size,
                    epochs=epochs,
                    sigma=sigma,
                )

            if curve_type == "gaussian":
                sigma = (1 / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
                budget = GaussianCurve(sigma=sigma)

            if curve_type == "laplace":
                budget = LaplaceCurve(laplace_noise=1 / epsilon)

            task_dict = {
                "alphas": budget.alphas,
                # "rdp_epsilons": list(map(lambda x: float(x), budget.epsilons)),
                "rdp_epsilons": np.array(budget.epsilons).tolist(),
                "n_blocks": f"{n_1}:0.5, {n_2}:0.5",
                "block_selection_policy": block_selection_policy,
            }

            task_name = f"{curve_type}_{epsilon}.yaml"
            task_names.append(task_name)
            yaml.dump(task_dict, tasks_path.joinpath(task_name).open("w"))

    logger.info(f"Saving the frequencies at {frequencies_path}...")
    frequencies_dict = {task_name: 1 / len(task_names) for task_name in task_names}
    yaml.dump(frequencies_dict, frequencies_path.joinpath("frequencies.yaml").open("w"))
    logger.info("Done.")


if __name__ == "__main__":
    typer.run(main)
