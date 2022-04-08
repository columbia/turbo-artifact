import random
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
from privacypacking.utils.zoo import (
    alpha_variance_frequencies,
    build_synthetic_zoo,
    build_zoo,
    gaussian_block_distribution,
    geometric_frequencies,
    zoo_df,
)

DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent.parent.joinpath("data")
# P_GRID = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
P_GRID = [0, 0.5, 1, 2, 4, 8]
app = typer.Typer()


@app.command()
def mixed(
    epsilon_1: float = typer.Option(0.5, help="Low budget tasks"),
    epsilon_2: float = typer.Option(1.0, help="High budget tasks"),
    delta: float = typer.Option(1e-7, help="Delta"),
    n_1: int = typer.Option(1, help="Low number of blocks"),
    n_2: int = typer.Option(10, help="High number of blocks"),
    p_1: float = typer.Option(1.0, help="Low profit"),
    p_2: float = typer.Option(1.0, help="High profit"),
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

            if p_1 != p_2:
                task_dict["profit"] = f"{p_1}:0.5, {p_2}:0.5"

            task_name = f"{curve_type}_{epsilon}.yaml"
            task_names.append(task_name)
            yaml.dump(task_dict, tasks_path.joinpath(task_name).open("w"))

    logger.info(f"Saving the frequencies at {frequencies_path}...")
    frequencies_dict = {task_name: 1 / len(task_names) for task_name in task_names}
    yaml.dump(frequencies_dict, frequencies_path.joinpath("frequencies.yaml").open("w"))
    logger.info("Done.")


@app.command()
def heterogeneous(
    # p: float = typer.Option(0.5, help="Poisson parameter for bin selection"),
    block_selection_policy: str = typer.Option(
        "RandomBlocks", help="Block selection policy"
    ),
    output_path: str = typer.Option(
        str(DEFAULT_OUTPUT_PATH.joinpath("heterogeneous_synthetic"))
    ),
    synthetic: bool = typer.Option(True),
):

    output_path = Path(output_path)

    tasks_path = output_path.joinpath("tasks")
    tasks_path.mkdir(exist_ok=True, parents=True)

    frequencies_path = output_path.joinpath("task_frequencies")
    frequencies_path.mkdir(exist_ok=True, parents=True)

    task_id_to_name = {}

    names_and_curves = build_synthetic_zoo() if synthetic else build_zoo()
    _, tasks_df = zoo_df(names_and_curves)

    for task_id in tasks_df.task_id:
        # for name, budget in names_and_curves:
        name, budget = names_and_curves[task_id]
        task_dict = {
            "alphas": budget.alphas,
            # "rdp_epsilons": list(map(lambda x: float(x), budget.epsilons)),
            "rdp_epsilons": np.array(budget.epsilons).tolist(),
            "n_blocks": f"1:1",
            # TODO: multiblock version!
            "block_selection_policy": block_selection_policy,
            "profit": "1:1",
        }

        task_name = f"{name}.yaml"
        task_id_to_name[task_id] = task_name
        yaml.dump(task_dict, tasks_path.joinpath(task_name).open("w"))

    mu = 10
    max_blocks = 20
    for sigma in [0, 1, 2, 4, 6, 10]:
        tasks_path = output_path.joinpath(f"tasks-mu{mu}-sigma{sigma}")
        tasks_path.mkdir(exist_ok=True, parents=True)
        for task_id in tasks_df.task_id:
            # for name, budget in names_and_curves:
            name, budget = names_and_curves[task_id]
            task_dict = {
                "alphas": budget.alphas,
                # "rdp_epsilons": list(map(lambda x: float(x), budget.epsilons)),
                "rdp_epsilons": np.array(budget.epsilons).tolist(),
                "n_blocks": gaussian_block_distribution(
                    mu=mu, sigma=sigma, max_blocks=max_blocks
                ),
                "block_selection_policy": block_selection_policy,
                "profit": "1:1",
            }

            task_name = f"{name}.yaml"
            yaml.dump(task_dict, tasks_path.joinpath(task_name).open("w"))

    logger.info(f"Saving the frequencies at {frequencies_path}...")

    for p in P_GRID:
        # p_tasks_df = geometric_frequencies(tasks_df, p=p)

        p_tasks_df = alpha_variance_frequencies(tasks_df, sigma=p)

        frequencies_dict = {}
        sum_frequencies = 0
        min_frequency = (
            1e-9  # Just ignore low probability tasks that can generate errors
        )
        for row in p_tasks_df.iterrows():
            task_id = int(row[1]["task_id"])
            frequency = float(row[1]["frequency"])

            if frequency > min_frequency:
                frequencies_dict[task_id_to_name[task_id]] = frequency
                sum_frequencies += frequency

        # Ensure that the frequencies add up to 1 even with rounding errors
        logger.info(f"Original sum: {sum_frequencies}")

        last_task_name, last_task_frequency = frequencies_dict.popitem()
        frequencies_dict[last_task_name] = last_task_frequency + 1 - sum_frequencies

        logger.info(f"Rectified to: {sum(frequencies_dict.values())}")

        yaml.dump(
            frequencies_dict,
            frequencies_path.joinpath(f"frequencies-{p}.yaml").open("w"),
        )

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


@app.command()
def privatekube(
    privacy_unit: str = typer.Option(
        default="event",
        help="Protection unit: `event`, `user`, `user-time`",
        show_default=True,
    ),
    gaussian_mice_fraction: float = typer.Option(
        default=0.0,
        help="Percentage of Gaussian mice to generate. The original PrivateKube workload does not use any.",
        show_default=True,
    ),
    laplace_mice_fraction: float = typer.Option(
        default=0.5,
        help="Percentage of Laplace mice to generate",
        show_default=True,
    ),
    profits: str = typer.Option(
        default="1",
        help="Use profit=1 by default, otherwise some mix. 'size' for p=eps*n_blocks. 'ksize' for int(1000*eps)*n_blocks. 'grid' for an arbitrary grid of profits.",
    ),
    block_policy: str = typer.Option(
        default="RandomBlocks",
        help="Block policy: `RandomBlocks`, `LatestBlocksFirst`, `Mix`",
        show_default=True,
    ),
    divide_original_block_number_by: int = typer.Option(
        default=100,
        help="The PrivateKube workload has block requests from 100 to 2000. 1 day of the Amazon data was split into 100 blocks, with 100 user id buckets.",
    ),
):

    workload_data_path = Path(__file__).parent.parent.parent.joinpath("data")

    privatekube_demands = workload_data_path.joinpath("privatekube_demands").joinpath(
        privacy_unit
    )

    output_dir = workload_data_path.joinpath(
        f"privatekube_{privacy_unit}_g{gaussian_mice_fraction}_l{laplace_mice_fraction}_p={profits}"
    )

    output_tasks_dir = output_dir.joinpath("tasks")
    output_tasks_dir.mkdir(exist_ok=True, parents=True)

    # Copy the task parameters and add profits/block policy
    task_category_names = {}
    for task_category in ["elephants", "mice-gaussian", "mice-laplace"]:
        task_category_names[task_category] = []
        for task_path in privatekube_demands.joinpath(task_category).glob("*.yaml"):
            task_name = f"{task_category}-{task_path.name}"
            raw_task_dict = yaml.safe_load(task_path.read_text())

            task_dict = {}
            task_dict["alphas"] = raw_task_dict["alphas"]
            task_dict["rdp_epsilons"] = raw_task_dict["rdp_epsilons"]
            task_dict["n_blocks"] = (
                raw_task_dict["n_blocks"] // divide_original_block_number_by
            )

            # WARNING: Important logic and arbitrary decisions in there!
            if profits == "grid":
                if task_category == "elephants":
                    profit = "500:0.25, 100:0.25, 50:0.25, 10:0.25"
                else:
                    profit = "50:0.25, 10:0.25, 5:0.25, 1:0.25"
            elif profits == "size":
                profit = raw_task_dict["epsilon"] * task_dict["n_blocks"]
            elif profits == "ksize":
                profit = int(1_000 * raw_task_dict["epsilon"]) * task_dict["n_blocks"]
            else:
                profit = 1

            if block_policy == "RandomBlocks":
                block_selection_policy = "RandomBlocks"
            elif block_policy == "LatestBlocksFirst":
                block_selection_policy = "LatestBlocksFirst"
            elif block_policy == "Mix":
                coin = random.randint(0, 1)
                if coin:
                    block_selection_policy = "RandomBlocks"
                else:
                    block_selection_policy = "LatestBlocksFirst"
            # END WARNING

            task_dict["profit"] = profit
            task_dict["block_selection_policy"] = block_selection_policy

            output_task_path = output_tasks_dir.joinpath(task_name)
            with open(output_task_path, "w") as f:
                yaml.dump(task_dict, f)

            task_category_names[task_category].append(task_name)

    logger.info(task_category_names)

    # List the tasks with frequencies
    task_frequencies = {}
    category_frequencies = {
        "mice-gaussian": gaussian_mice_fraction,
        "mice-laplace": laplace_mice_fraction,
        "elephants": 1 - gaussian_mice_fraction - laplace_mice_fraction,
    }
    sum_frequencies = 0
    for category, task_names in task_category_names.items():
        n = len(task_names)
        logger.info(f"{category}, {task_names}, {n}")

        for task_name in task_names:
            # cat_task_name = f"{category}-{task_name}"
            task_frequencies[task_name] = category_frequencies[category] / n
            sum_frequencies += task_frequencies[task_name]
    # Ensure that the frequencies add up to 1 even with rounding errors
    logger.info(sum_frequencies)
    last_task_name, last_task_frequency = task_frequencies.popitem()
    task_frequencies[last_task_name] = last_task_frequency + 1 - sum_frequencies
    logger.info(sum(list(task_frequencies.values())))
    logger.info(task_frequencies)
    output_frequencies_path = output_dir.joinpath("task_frequencies").joinpath(
        "frequencies.yaml"
    )
    output_frequencies_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_frequencies_path, "w") as f:
        yaml.dump(task_frequencies, f)


if __name__ == "__main__":
    app()
