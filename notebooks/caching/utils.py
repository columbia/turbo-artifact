import pandas as pd
import plotly.express as px
from plotly.offline import iplot
from precycle.utils.utils import LOGS_PATH
from experiments.ray.analysis import load_ray_experiment


def get_df(path):
    logs = LOGS_PATH.joinpath("ray/" + path)
    df = load_ray_experiment(logs)
    return df


def plot_num_allocated_tasks(df):
    total_tasks = df["total_tasks"].max()
    query_pool_sizes_order = [
        str(x) for x in sorted(df["query_pool_size"].astype(int).unique())
    ]

    fig = px.bar(
        df,
        x="query_pool_size",
        y="n_allocated_tasks",
        color="key",
        barmode="group",
        title=f"Num allocated tasks - total tasks={total_tasks}",
        width=900,
        height=500,
        category_orders={"query_pool_size": query_pool_sizes_order},
    )
    return fig


def plot_errors(df):
    return px.scatter(
        df,
        x="n_blocks",
        y="error",
        color="key",
        title="Error wrt to the ground truth",
    )


def plot_results(df):
    return px.scatter(
        df,
        x="n_blocks",
        y="result",
        color="key",
        title="Result",
    )


def plot_planning_time(df):
    return px.scatter(
        df,
        x="n_blocks",
        y="planning_time",
        color="key",
        title="Planning time",
    )


def plot_budget_utilization(df):
    df["budget_utilization"] = (df["initial_budget"] - df["budget"]) / df[
        "initial_budget"
    ]
    query_pool_sizes_order = [
        str(x) for x in sorted(df["query_pool_size"].astype(int).unique())
    ]

    fig = px.bar(
        df,
        x="query_pool_size",
        y="budget_utilization",
        color="key",
        barmode="group",
        title="Budget Utilization",
        width=900,
        height=500,
        category_orders={"query_pool_size": query_pool_sizes_order},
        color_discrete_map={
            "DeterministicCache": "blue",
            "ProbabilisticCache": "green",
        },
        range_y=[0, 1],
    )
    return fig


def plot_hard_queries(df):
    query_pool_sizes_order = [
        str(x) for x in sorted(df["query_pool_size"].astype(int).unique())
    ]

    fig = px.bar(
        df,
        x="query_pool_size",
        y="hard_queries",
        color="key",
        barmode="group",
        title="Hard Queries",
        width=900,
        height=500,
        category_orders={"query_pool_size": query_pool_sizes_order},
        color_discrete_map={
            "DeterministicCache": "blue",
            "ProbabilisticCache": "green",
        },
    )
    return fig


def get_tasks_information(df):
    dfs = []

    for (tasks, key, query_pool_size) in df[
        ["tasks_info", "key", "query_pool_size"]
    ].values:
        for task in tasks:
            dfs.append(
                pd.DataFrame(
                    [
                        {
                            "id": task["id"],
                            "result": task["result"],
                            # "error": task["error"],
                            "planning_time": task["planning_time"],
                            "blocks": task["blocks"],
                            "n_blocks": task["n_blocks"],
                            "hard_query": task["run_metadata"]["hard_query"],
                            "utility": task["utility"],
                            "utility_beta": task["utility_beta"],
                            "status": task["status"],
                            "key": key,
                            "query_pool_size": query_pool_size,
                        }
                    ]
                )
            )
    if dfs:
        tasks = pd.concat(dfs)
        tasks["result"] = tasks["result"].astype(float)
        # tasks["error"] = tasks["error"].astype(float)
        tasks["planning_time"] = tasks["planning_time"].astype(float)
        return tasks
    return None


def get_blocks_information(df):
    dfs = []

    for (blocks, initial_budget, key, query_pool_size) in df[
        ["block_budgets_info", "blocks_initial_budget", "key", "query_pool_size"]
    ].values:
        for block_id, budget in blocks:
            orders = budget["orders"]
            order = max(orders, key=orders.get)
            max_available_budget = orders[order]
            max_initial_budget = initial_budget["orders"][order]
            dfs.append(
                pd.DataFrame(
                    [
                        {
                            "id": block_id,
                            "initial_budget": max_initial_budget,
                            "budget": max_available_budget,
                            "key": key,
                            "query_pool_size": query_pool_size,
                        }
                    ]
                )
            )
    if dfs:
        blocks = pd.concat(dfs)
        return blocks
    return None


def analyze_experiment(experiment_path):
    df = get_df(experiment_path)
    # print(df.keys())
    key_columns = ["cache"]

    df["key"] = " "
    for col in key_columns:
        df["key"] += df[col] + " "

    df["query_pool_size"] = df["query_pool_size"].astype(str)

    df.sort_values(["key"], ascending=[True], inplace=True)

    # tasks = get_tasks_information(df)
    blocks = get_blocks_information(df)

    iplot(plot_num_allocated_tasks(df))
    iplot(plot_budget_utilization(blocks))
    iplot(plot_hard_queries(df))
    # iplot(plot_errors(tasks))
    # iplot(plot_results(tasks))
    # iplot(plot_planning_time(tasks))


#     iplot(plot_budget_utilization_total(blocks))
