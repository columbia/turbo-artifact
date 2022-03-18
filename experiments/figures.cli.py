import os
from pathlib import Path

import plotly.express as px
import typer
from loguru import logger
from ray import tune

from experiments.ray.analysis import load_ray_experiment
from experiments.ray_runner import grid_offline, grid_online

app = typer.Typer()


def map_metric_to_id(row):
    d = {
        "DominantShares": 0,
        "FlatRelevance": 1,
        "OverflowRelevance": 2,
        "ArgmaxKnapsack": 3,
        "Simplex": 4,
    }
    return d[row]


# TODO: trigger make directly?
# TODO: custom option to overwrite makefiles

# TODO: reuse the plotting functions


def plot_3a(fig_dir):
    experiment_analysis = grid_offline(
        custom_config="offline_dpf_killer/multi_block/gap_base.yaml",
        num_blocks=[5, 10, 15, 20],
        num_tasks=[100],
        data_path="multiblock_dpf_killer_gap",
        parallel=False,
    )

    all_trial_paths = experiment_analysis._get_trial_paths()
    experiment_dir = Path(all_trial_paths[0]).parent

    rdf = load_ray_experiment(experiment_dir)
    # rdf["scheduler_metric"] = rdf.apply(
    #     lambda row: row.scheduler_metric
    #     if row.scheduler == "basic_scheduler"
    #     else "Simplex",
    #     axis=1,
    # )

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


def plot_3b(fig_dir):
    experiment_analysis = grid_offline(
        custom_config="offline_dpf_killer/single_block/base.yaml",
        num_blocks=[1],
        num_tasks=[1] + [5 * i for i in range(1, 6)],
        data_path="single_block_dpf_killer_subsampled",
    )

    all_trial_paths = experiment_analysis._get_trial_paths()
    experiment_dir = Path(all_trial_paths[0]).parent

    rdf = load_ray_experiment(experiment_dir)
    # rdf["scheduler_metric"] = rdf.apply(
    #     lambda row: row.scheduler_metric
    #     if row.scheduler == "basic_scheduler"
    #     else "Simplex",
    #     axis=1,
    # )

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


def plot_4(fig_dir):
    experiment_analysis = grid_offline(
        custom_config="offline_dpf_killer/multi_block/gap_base.yaml",
        num_blocks=[20],
        num_tasks=[50, 100, 200, 300, 350, 400, 500, 750, 1000, 1500, 2000],
        data_path="mixed_curves",
        metric_recomputation_period=100,
        parallel=False,  # We care about the runtime here
    )

    all_trial_paths = experiment_analysis._get_trial_paths()
    experiment_dir = Path(all_trial_paths[0]).parent

    rdf = load_ray_experiment(experiment_dir)
    rdf["scheduler_metric"] = rdf.apply(
        lambda row: row.scheduler_metric
        if row.scheduler == "basic_scheduler"
        else "Simplex",
        axis=1,
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


def plot_5(fig_dir):

    experiment_analysis = grid_online(
        custom_config="time_based_budget_unlocking/privatekube/base.yaml"
    )

    logger.info(experiment_analysis)

    # raise NotImplementedError(
    #     "We're not sure we'll keep this figure in the final paper yet."
    # )


def plot_6(fig_dir):
    raise NotImplementedError(
        "We're not sure we'll keep this figure in the final paper yet."
    )


def plot_7(fig_dir):
    raise NotImplementedError(
        "This CLI only works for the simulator, not the real PrivateKube system."
    )


def plot_fairness(fig_dir):
    raise NotImplementedError(
        """We will probably plot this on the Alibaba workload. But for reference, check out the following files:
            - experiments/ray/offline_privatekube/fair_tasks.py
            - experiments/ray/offline_mixed_curves/monoalpha_fairness.py
            - /home/pierre/privacypacking/notebooks/offline_mixed_curves/privatekube_fair_tasks.ipynb
            - notebooks/offline_mixed_curves/multialpha_fair_tasks.ipynb
        """
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

    # rdf = load_ray_experiment(
    #     Path("/home/pierre/privacypacking/logs/ray/DEFAULT_2022-03-02_11-03-24")
    # )

    # print(rdf)
    # print(rdf.columns)
    # rdf.sort_values("n_initial_blocks")
