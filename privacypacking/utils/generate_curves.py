from pathlib import Path

import numpy as np
import typer
import yaml
from loguru import logger

from privacypacking.budget import ALPHAS, Budget
from privacypacking.budget.curves import (
    GaussianCurve,
    LaplaceCurve,
    SubsampledGaussianCurve,
)
from privacypacking.budget.utils import compute_noise_from_target_epsilon

DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent.parent.joinpath("data")

app = typer.Typer()


@app.command()
def mixed(
    epsilon_1: float = typer.Option(0.5, help="Low budget tasks"),
    epsilon_2: float = typer.Option(1.0, help="High budget tasks"),
    delta: float = typer.Option(1e-7, help="Delta"),
    n_1: int = typer.Option(1, help="Low number of blocks"),
    n_2: int = typer.Option(10, help="High number of blocks"),
    block_selection_policy: str = typer.Option(
        "RandomBlocks", help="Block selection policy"
    ),
    output_path: str = typer.Option(str(DEFAULT_OUTPUT_PATH.joinpath("mixed_curves"))),
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
                batch_size = 50
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


@app.command()
def demo(
    epsilon: float = typer.Option(
        10.0, help="Block epsilon. Check the PrivateKube defaults."
    ),
    delta: float = typer.Option(1e-5, help="Block delta"),
    flat_fraction_low: float = typer.Option(1 / 3, help="Low budget flat tasks"),
    flat_fraction_high: float = typer.Option(1 / 2, help="High budget flat tasks"),
    bumpy_fraction_low: float = typer.Option(1 / 5, help="Low budget bumpy tasks"),
    bumpy_fraction_high: float = typer.Option(1 / 4, help="High budget bumpy tasks"),
    bumpy_step: float = typer.Option(
        1.0, help="Step between orders for the bumpy tasks"
    ),
    n_1: int = typer.Option(1, help="Low number of blocks"),
    n_2: int = typer.Option(2, help="High number of blocks"),
    block_selection_policy: str = typer.Option(
        "RandomBlocks", help="Block selection policy"
    ),
    output_path: str = typer.Option(str(DEFAULT_OUTPUT_PATH.joinpath("demo_workload"))),
    privatekube_output: bool = typer.Option(
        False,
        help="Create subdirs for each type of task and scale the demands for PrivateKube",
    ),
    block_multiplicative_factor: int = typer.Option(
        100, help="Multiply the demands, if privatekube output."
    ),
):
    output_path = Path(output_path)

    if privatekube_output:
        elephants_path = output_path.joinpath("elephants")
        mice_path = output_path.joinpath("mice")
        elephants_path.mkdir(exist_ok=True, parents=True)
        mice_path.mkdir(exist_ok=True, parents=True)
    else:
        tasks_path = output_path.joinpath("tasks")
        tasks_path.mkdir(exist_ok=True, parents=True)

        frequencies_path = output_path.joinpath("task_frequencies")
        frequencies_path.mkdir(exist_ok=True, parents=True)

    task_names = []
    block_budget = Budget.from_epsilon_delta(epsilon=epsilon, delta=delta)

    def make_bumpy_budget(start, step=1, n_flat_alphas=6):
        orders = {}
        i = 0
        for alpha, epsilon in zip(block_budget.alphas, block_budget.epsilons):
            eps = epsilon * start * (1 + max(i - n_flat_alphas, 0) * step)
            # TODO: make sure we handle empty initial epsilon properly in other tasks too
            orders[alpha] = eps if eps > 0 else 0.00001
            i += 1
        return Budget(orders)

    task_budgets = {
        "short_flat": make_bumpy_budget(flat_fraction_high, n_flat_alphas=20),
        "long_flat": make_bumpy_budget(flat_fraction_low, n_flat_alphas=20),
        "short_bumpy": make_bumpy_budget(bumpy_fraction_high, step=bumpy_step),
        "long_bumpy": make_bumpy_budget(bumpy_fraction_low, step=bumpy_step),
    }

    for task_name, task_budget in task_budgets.items():

        task_names.append(f"{task_name}.yaml")
        task_dict = {
            "alphas": task_budget.alphas,
            "rdp_epsilons": np.array(task_budget.epsilons).tolist(),
            "n_blocks": n_1 if "short" in task_name else n_2,
            "block_selection_policy": block_selection_policy,
        }

        if privatekube_output:
            task_dict["n_blocks"] *= block_multiplicative_factor
            if "flat" in task_name:
                yaml.dump(task_dict, mice_path.joinpath(f"{task_name}.yaml").open("w"))
            else:
                yaml.dump(
                    task_dict, elephants_path.joinpath(f"{task_name}.yaml").open("w")
                )

        else:
            yaml.dump(task_dict, tasks_path.joinpath(f"{task_name}.yaml").open("w"))

    if not privatekube_output:
        logger.info(f"Saving the frequencies at {frequencies_path}...")
        frequencies_dict = {task_name: 1 / len(task_names) for task_name in task_names}
        yaml.dump(
            frequencies_dict, frequencies_path.joinpath("frequencies.yaml").open("w")
        )
    logger.info("Done.")


if __name__ == "__main__":
    app()
