import os
from pathlib import Path

import plotly.express as px
import typer
from loguru import logger
from ray import tune

from experiments.ray_runner import (
    grid_offline,
    grid_offline_heterogeneity_knob,
    grid_online,
)

app = typer.Typer()


def map_metric_to_id(row):
    d = {
        "DominantShares": 0,
        "FlatRelevance": 1,
        "OverflowRelevance": 2,
        "ArgmaxKnapsack": 3,
        "simplex": 4,
    }
    return d[row]


# TODO: trigger make directly?
# TODO: custom option to overwrite makefiles

# TODO: reuse the plotting functions


def plot_3a(fig_dir):
    plot_multiblock_dpf_killer(fig_dir)


def plot_4(fig_dir):
    plot_mixed_curves_offline(fig_dir)


def plot_5(fig_dir):
    plot_mixed_curves_online(fig_dir)


def plot_6(fig_dir):
    plot_alibaba(fig_dir)


def plot_7(fig_dir):
    raise NotImplementedError(
        "This CLI only works for the simulator, not the real PrivateKube system."
    )


def plot_multiblock_dpf_killer(fig_dir):
    rdf = grid_offline(
        num_blocks=[5, 10, 15, 20],
        num_tasks=[100],
        data_path=["multiblock_dpf_killer_gap"],
        parallel=False,
    )

    fig = px.line(
        rdf.sort_values("n_initial_blocks"),
        x="n_initial_blocks",
        y="n_allocated_tasks",
        color="scheduler_metric",
        width=800,
        height=600,
        range_y=[0, 50],
        title="DPF inefficiency example 1",
    )

    gnuplot_df = rdf
    gnuplot_df["id"] = gnuplot_df.scheduler_metric.apply(map_metric_to_id)
    gnuplot_df = (
        gnuplot_df[
            [
                "n_initial_blocks",
                "n_allocated_tasks",
                "id",
                "scheduler",
                "scheduler_metric",
            ]
        ]
        .sort_values(["id", "n_initial_blocks"])
        .drop_duplicates()
    )

    fig_path = fig_dir.joinpath("motivating_examples_simulation/problem_1.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(fig_path)

    gnuplot_df.to_csv(
        fig_path.with_suffix(".csv"),
        index=False,
    )


def plot_single_block_dpf_killer(fig_dir):
    rdf = grid_offline(
        num_blocks=[1],
        num_tasks=[1] + [5 * i for i in range(1, 6)],
        data_path=["single_block_dpf_killer_subsampled"],
    )

    fig = px.line(
        rdf.sort_values("total_tasks"),
        x="total_tasks",
        y="n_allocated_tasks",
        color="scheduler_metric",
        width=800,
        height=600,
        range_y=[0, 25],
        title="Diverse RDP curves offline",
    )

    gnuplot_df = rdf
    gnuplot_df["id"] = gnuplot_df.scheduler_metric.apply(map_metric_to_id)
    gnuplot_df = (
        gnuplot_df[
            [
                "total_tasks",
                "n_allocated_tasks",
                "id",
                "scheduler",
                "scheduler_metric",
            ]
        ]
        .sort_values(["id", "total_tasks"])
        .drop_duplicates()
    )

    fig_path = fig_dir.joinpath("motivating_examples_simulation/problem_2.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(fig_path)

    gnuplot_df.to_csv(
        fig_path.with_suffix(".csv"),
        index=False,
    )


def plot_mixed_curves_offline(fig_dir):
    rdf = grid_offline(
        num_blocks=[20],
        # num_tasks=[50, 100, 200, 300, 350, 400, 500, 750, 1000, 1500, 2000],
        # num_tasks=[50, 100, 200, 300, 500],
        num_tasks=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
        data_path=["mixed_curves"],
        metric_recomputation_period=100,
        parallel=False,  # We care about the runtime here
        gurobi_timeout_minutes=1,
    )

    fig = px.line(
        rdf.sort_values("total_tasks"),
        x="total_tasks",
        y="n_allocated_tasks",
        color="scheduler_metric",
        width=800,
        height=600,
        range_y=[0, 1500],
        title="Diverse RDP curves offline",
    )

    gnuplot_df = rdf
    gnuplot_df["id"] = gnuplot_df.scheduler_metric.apply(map_metric_to_id)
    gnuplot_df = (
        gnuplot_df[
            [
                "total_tasks",
                "n_allocated_tasks",
                "id",
                "scheduler",
                "scheduler_metric",
            ]
        ]
        .sort_values(["id", "total_tasks"])
        .drop_duplicates()
    )

    fig_path = fig_dir.joinpath(
        "offline_mixed_curves/offline_mixed_curves_without_profits.png"
    )
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(fig_path)

    gnuplot_df.to_csv(
        fig_path.with_suffix(".csv"),
        index=False,
    )

    # Let's plot the runtime now

    fig = px.line(
        rdf.sort_values("total_tasks"),
        x="total_tasks",
        y="time_total_s",
        color="scheduler_metric",
        # log_x=True,
        width=800,
        height=600,
        range_y=[0, 1_000],
        # title="Number of allocated tasks depending on the scheduling step size<br><sup>Online mixed curves, 20 blocks, no initial blocks, 100 tasks per block on average, lifetime = 5 blocks</sup>"
    )

    gnuplot_df = rdf
    gnuplot_df["id"] = gnuplot_df.scheduler_metric.apply(map_metric_to_id)
    gnuplot_df = (
        gnuplot_df[
            [
                "total_tasks",
                "time_total_s",
                "id",
                "scheduler",
                "scheduler_metric",
            ]
        ]
        .sort_values(["id", "total_tasks"])
        .drop_duplicates()
    )
    fig_path = fig_dir.joinpath(
        "offline_mixed_curves/offline_mixed_curves_without_profits_runtime.png"
    )
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(fig_path)

    gnuplot_df.to_csv(
        fig_path.with_suffix(".csv"),
        index=False,
    )


def plot_mixed_curves_online(fig_dir):
    rdf = grid_online(
        scheduler_scheduling_time=[2, 4, 6, 8, 10],
        metric_recomputation_period=[50],
        initial_blocks=[10],
        max_blocks=[30],
        data_path=["mixed_curves"],
        tasks_sampling="poisson",
        data_lifetime=[10],
        avg_num_tasks_per_block=[500],
    )

    fig = px.line(
        rdf.sort_values("T"),
        x="T",
        y="n_allocated_tasks",
        color="scheduler_metric",
        width=800,
        height=600,
        range_y=[0, 1500],
        title="Mixed curves online",
    )

    fig_path = fig_dir.joinpath("mixed-curves/online.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(fig_path)


# TODO: remember to update delta if you are using something else than mixed curves.


def plot_fairness(fig_dir):
    raise NotImplementedError(
        """We will probably plot this on the Alibaba workload. But for reference, check out the following files:
            - experiments/ray/offline_privatekube/fair_tasks.py
            - experiments/ray/offline_mixed_curves/monoalpha_fairness.py
            - /home/pierre/privacypacking/notebooks/offline_mixed_curves/privatekube_fair_tasks.ipynb
            - notebooks/offline_mixed_curves/multialpha_fair_tasks.ipynb
        """
    )


def plot_alibaba(fig_dir):
    rdf = grid_online(
        scheduler_scheduling_time=[0.01, 0.1, 1, 10],
        metric_recomputation_period=[50],
        initial_blocks=[10],
        max_blocks=[50],
        data_path=["alibaba-privacy-workload/outputs/privacy_tasks.csv"],
        tasks_sampling="",
        data_lifetime=[10],
    )

    fig = px.line(
        rdf.sort_values("T"),
        x="T",
        y="n_allocated_tasks",
        color="scheduler_metric",
        width=800,
        height=600,
        range_y=[0, 1500],
        title="Alibaba",
    )

    fig_path = fig_dir.joinpath("alibaba/alibaba.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(fig_path)


def plot_temp(fig_dir):
    rdf = grid_offline_heterogeneity_knob(
        num_blocks=[20],
        # num_tasks=[50, 100, 200, 300, 350, 400, 500, 750, 1000, 1500, 2000],
        num_tasks=[10_000],
        # num_tasks=[20_000],
        data_path="heterogeneous",
        metric_recomputation_period=100,
        parallel=False,  # We care about the runtime here
        gurobi_timeout_minutes=1,
    )

    fig = px.line(
        # rdf.sort_values("variance"),
        # x="variance",
        rdf.sort_values("block_std"),
        x="block_std",
        y="n_allocated_tasks",
        color="scheduler_metric",
        width=800,
        height=600,
        log_x=True,
        # range_y=[0, 3000],
        title="Heterogeneous RDP curves offline",
    )
    fig.update_yaxes(rangemode="tozero")

    gnuplot_df = rdf
    gnuplot_df["id"] = gnuplot_df.scheduler_metric.apply(map_metric_to_id)
    gnuplot_df = (
        gnuplot_df[
            [
                "total_tasks",
                "n_allocated_tasks",
                "id",
                "scheduler",
                "scheduler_metric",
            ]
        ]
        .sort_values(["id", "total_tasks"])
        .drop_duplicates()
    )

    fig_path = fig_dir.joinpath("temp.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(fig_path)

    gnuplot_df.to_csv(
        fig_path.with_suffix(".csv"),
        index=False,
    )


@app.command()
def run(
    fig: str = "3a",
    loguru_level: str = "WARNING",
    save_csv: bool = True,
    fig_dir: str = "",
):
    """
    Command line interface to reproduce the figures from the paper.
    Usage:
    `python experiments/figures.cli.py --fig 4`

    Fig can be any of the following:
        3a, 3b, 4, fairness

    Run `python experiments/figures.cli.py --help` for more information.
    """

    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    if not fig_dir:
        fig_dir = Path(__file__).parent.joinpath("figures")
    else:
        fig_dir = Path(fig_dir)
    # fig_dir.mkdir(parents=True, exist_ok=True)

    globals()[f"plot_{fig}"](fig_dir)

    # if fig == "3a":
    #     plot_3a(fig_dir)
    # elif fig == "3b":
    #     plot_3b(fig_dir)


if __name__ == "__main__":
    app()
