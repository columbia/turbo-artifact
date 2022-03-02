import os
from pathlib import Path

import plotly.express as px
import typer
from ray import tune

from experiments.ray.analysis import load_ray_experiment
from experiments.ray_runner import grid_offline

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


def plot_3a(fig_dir):
    experiment_analysis = grid_offline(
        custom_config="offline_dpf_killer/multi_block/gap_base.yaml",
        num_blocks=[5, 10, 15, 20],
        num_tasks=[100],
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
        range_y=[0, 25],
        title="DPF inefficiency example 2",
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


@app.command()
def run(
    fig: str = "3a",
    loguru_level: str = "WARNING",
    save_csv: bool = True,
    fig_dir: str = "",
):

    os.environ["LOGURU_LEVEL"] = "WARNING"
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    if not fig_dir:
        fig_dir = Path(__file__).parent.joinpath("figures")
    else:
        fig_dir = Path(fig_dir)
    # fig_dir.mkdir(parents=True, exist_ok=True)

    if fig == "3a":
        plot_3a(fig_dir)
    elif fig == "3b":
        plot_3b(fig_dir)


if __name__ == "__main__":
    app()

    # rdf = load_ray_experiment(
    #     Path("/home/pierre/privacypacking/logs/ray/DEFAULT_2022-03-02_11-03-24")
    # )

    # print(rdf)
    # print(rdf.columns)
    # rdf.sort_values("n_initial_blocks")
