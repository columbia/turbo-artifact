import random
from collections import defaultdict
from pathlib import Path

import typer
import yaml
from loguru import logger


def main(
    privacy_unit: str = typer.Option(
        default="event",
        help="Protection unit: `event`, `user`, `user-time`",
        show_default=True,
    ),
    gaussian_mice_fraction: float = typer.Option(
        default=0.3,
        help="Percentage of Gaussian mice to generate",
        show_default=True,
    ),
    laplace_mice_fraction: float = typer.Option(
        default=0.3,
        help="Percentage of Laplace mice to generate",
        show_default=True,
    ),
    profits: bool = typer.Option(
        default=False, help="Use profit=1 by default, otherwise some mix"
    ),
    block_policy: str = typer.Option(
        default="RandomBlocks",
        help="Block policy: `RandomBlocks`, `LatestBlocksFirst`, `Mix`",
        show_default=True,
    ),
    divide_original_block_number_by: int = typer.Option(
        default=10,
        help="The PrivateKube workload has block requests from 100 to 2000. 1 day of the Amazon data was split into 100 blocks, with 100 user id buckets.",
    ),
):

    privatekube_demands = (
        Path(__file__).parent.joinpath("privatekube_demands").joinpath(privacy_unit)
    )

    output_dir = Path(__file__).parent.joinpath(
        f"privatekube_{privacy_unit}_g{gaussian_mice_fraction}_l{laplace_mice_fraction}_{'p!=1' if profits else 'p=1'}"
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
            if profits:
                if task_category == "elephants":
                    profit = "10:0.5, 5:0.5"
                else:
                    profit = "2:0.5, 1:0.5"
            else:
                profit = 1

            if block_policy == "RandomBlocks":
                block_selection_policy = "RandomBlocks"
            elif block_policy == "LatestBlocksFirst":
                block_selection_policy = "LatestBlocksFirst"
            elif block_policy == "Mix":
                # TODO: doesn't quite make sense for statistics. Implement static contiguous range too?
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
    typer.run(main)
