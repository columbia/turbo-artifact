import pandas as pd
import plotly.express as px
from plotly.offline import iplot
from privacypacking.utils.utils import LOGS_PATH
from experiments.ray.analysis import load_ray_experiment


def get_df(path):
    logs = LOGS_PATH.joinpath("ray/" + path)
    df = load_ray_experiment(logs)
    return df


def plot_num_allocated_tasks(df):
    total_tasks = df["total_tasks"].max()
    return px.scatter(
        df,
        x="key",
        y="n_allocated_tasks",
        color="key",
        title=f"Num allocated tasks - total tasks={total_tasks}",
    )


def plot_errors(df):
    return px.scatter(
        df,
        x="num_requested_blocks",
        y="error",
        color="key",
        title="Error wrt to the ground truth",
    )


def plot_results(df):
    return px.scatter(
        df,
        x="num_requested_blocks",
        y="result",
        color="key",
        title="Result",
    )


def plot_planning_time(df):
    return px.scatter(
        df,
        x="num_requested_blocks",
        y="planning_time",
        color="key",
        title="Planning time",
    )


def plot_budget_utilization(df):
    df["budget_utilization"] = (df["initial_budget"] - df["budget"]) / df[
        "initial_budget"
    ]
    return px.bar(
        df,
        x="id",
        y="budget_utilization",
        color="key",
        title="Budget Utilization",
        range_y=[0, 1],
    )

def plot_budget_utilization_total(df):
    # df["budget_utilization_total"] = (df["initial_budget"] - df["budget"]).sum() / df[
        # "initial_budget"
    # ].sum()
    groups = df.groupby("key")[['initial_budget', 'budget']].sum()
    groups['budget_utilization_total'] = (groups['initial_budget']-groups['budget'])/groups['initial_budget']
    return px.bar(
        groups.reset_index(),
        x="key",
        y="budget_utilization_total",
        color="key",
        title="Total Budget Utilization",
        range_y=[0, 1],
    )


def get_tasks_information(df):
    dfs = []
    for (tasks, key) in df[["tasks", "key"]].values:
        for task in tasks:
            dfs.append(
                pd.DataFrame(
                    [
                        {
                            "id": task["id"],
                            "result": task["result"],
                            "error": task["error"],
                            "planning_time": task["planning_time"],
                            "num_requested_blocks": task["num_blocks"],
                            "key": key,
                        }
                    ]
                )
            )
    if dfs:
        tasks = pd.concat(dfs)
        tasks["result"] = tasks["result"].astype(float)
        tasks["error"] = tasks["error"].astype(float)
        tasks["planning_time"] = tasks["planning_time"].astype(float)
        return tasks
    return None


def get_blocks_information(df):
    dfs = []
    for (blocks, key) in df[["blocks", "key"]].values:
        for block in blocks:
            # orders = block['budget']['orders']
            # order = max(orders, key=orders.get)
            # max_available_budget = orders[order]
            # initial_budget = block['initial_budget']['orders'][order]
            # dfs.append(pd.DataFrame([{"id": block['id'],
            #                           "initial_budget": initial_budget,
            #                           "budget": max_available_budget,
            #                           "key": key,
            #                         }]))
            dfs.append(
                pd.DataFrame(
                    [
                        {
                            "id": block["id"],
                            "initial_budget": block["initial_budget"]["epsilon"],
                            "budget": block["budget"]["epsilon"],
                            "key": key,
                        }
                    ]
                )
            )
    if dfs:
        blocks = pd.concat(dfs)
        return blocks
    return None


def analyze_experiment(tasks_path, experiment_path):
    tasks = pd.read_csv(tasks_path)
    tasks.reset_index()
    tasks["id"] = tasks.index

    fig = px.scatter(
        tasks,
        x="id",
        y="submit_time",
        title="arrival time per task",
    )
    iplot(fig)

    fig = px.scatter(
        tasks,
        x="id",
        y="n_blocks",
        title="Requested numbers of Blocks per task",
    )
    iplot(fig)

    df = get_df(experiment_path)
    df["utility"] = df["utility"].astype(str)
    # df["variance_reduction"]=df["variance_reduction"].astype(str)
    df = df.astype({'variance_reduction': 'str'})
    df["key"] = df["planner"] + " " + df["utility"] + " " + df["optimization_objective"] + " VR=" + df["variance_reduction"]
    df.drop(columns=["planner", "cache"], inplace=True)

    metrics = df[["key", "n_allocated_tasks", "total_tasks"]]
    df = df[["key", "tasks", "blocks"]]

    tasks = get_tasks_information(df)
    tasks = tasks.groupby(["id", "key", "num_requested_blocks"]).mean().reset_index()

    blocks = get_blocks_information(df)
    blocks = (
        blocks.groupby(["id", "key", "budget", "initial_budget"]).mean().reset_index()
    )

    iplot(plot_num_allocated_tasks(metrics))
    iplot(plot_errors(tasks))
    iplot(plot_results(tasks))
    iplot(plot_planning_time(tasks))
    iplot(plot_budget_utilization(blocks))
    iplot(plot_budget_utilization_total(blocks))

