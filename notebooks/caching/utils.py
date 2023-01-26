import pandas as pd
import plotly.express as px
from plotly.offline import iplot
from precycle.utils.utils import LOGS_PATH
from experiments.ray.analysis import load_ray_experiment


def get_df(path):
    logs = LOGS_PATH.joinpath("ray/" + path)
    df = load_ray_experiment(logs)
    return df


def get_tasks_information(df):
    dfs = []

    for (tasks, key, query_pool_size, heuristic_threshold) in df[
        ["tasks_info", "key", "query_pool_size", "heuristic_threshold"]
    ].values:

        for task in tasks:

            deterministic_runs = task["deterministic_runs"]
            probabilistic_runs = task["probabilistic_runs"]

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
                            # "hard_run_ops": task["run_metadata"]["hard_run_ops"],
                            "utility": task["utility"],
                            "utility_beta": task["utility_beta"],
                            "status": task["status"],
                            "key": key,
                            "query_pool_size": query_pool_size,
                            "heuristic_threshold": heuristic_threshold,
                            "deterministic_runs": deterministic_runs,
                            "probabilistic_runs": probabilistic_runs,
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

    for (blocks, initial_budget, key, query_pool_size, past_queries_len) in df[
        [
            "block_budgets_info",
            "blocks_initial_budget",
            "key",
            "query_pool_size",
            "past_queries_len",
        ]
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
                            "past_queries_len": past_queries_len,
                        }
                    ]
                )
            )
    if dfs:
        blocks = pd.concat(dfs)
        return blocks
    return None


def analyze_experiment1(experiment_path):
    def plot_num_allocated_tasks(df, total_tasks):
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

    def plot_budget_utilization(df, total_tasks):
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
            title=f"Budget Utilization - total tasks-{total_tasks}",
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

    def plot_hard_run_ops(df, total_tasks):
        query_pool_sizes_order = [
            str(x) for x in sorted(df["query_pool_size"].astype(int).unique())
        ]
        fig = px.bar(
            df,
            x="query_pool_size",
            y="avg_total_hard_run_ops",
            color="key",
            barmode="group",
            title=f"Hard Queries - total tasks={total_tasks}",
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

    df = get_df(experiment_path)
    df["query_pool_size"] = df["query_pool_size"].astype(str)
    df["key"] = df["cache"]
    df.sort_values(["key"], ascending=[True], inplace=True)
    total_tasks = df["total_tasks"].max()
    blocks = get_blocks_information(df)
    iplot(plot_num_allocated_tasks(df, total_tasks))
    iplot(plot_budget_utilization(blocks, total_tasks))
    iplot(plot_hard_run_ops(df, total_tasks))


def analyze_experiment2(experiment_path):
    def plot_budget_utilization(df, total_tasks):
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
            title=f"Budget Utilization - total tasks-{total_tasks}",
            width=900,
            height=500,
            range_y=[0, 1],
            category_orders={"query_pool_size": query_pool_sizes_order},
        )
        return fig

    def plot_hard_run_ops(df, total_tasks):
        query_pool_sizes_order = [
            str(x) for x in sorted(df["query_pool_size"].astype(int).unique())
        ]
        fig = px.bar(
            df,
            x="query_pool_size",
            y="avg_total_hard_run_ops",
            color="key",
            barmode="group",
            title=f"Hard Queries - total tasks={total_tasks}",
            width=900,
            height=500,
            range_y=[0, 1],
            category_orders={"query_pool_size": query_pool_sizes_order},
        )
        return fig

    df = get_df(experiment_path)
    df = df.drop(columns=["tasks_info"])  # Make it lighter
    total_tasks = df["total_tasks"].max()

    df.sort_values(["past_queries_len"], ascending=[True], inplace=True)
    df["query_pool_size"] = df["query_pool_size"].astype(str)
    df["avg_bin_visits"] = df["avg_bin_visits"].astype(str)
    df["past_queries_len"] = df["past_queries_len"].astype(str)

    heuristic = df["heuristic"].max()
    df["key"] = (
        df["cache"] + "-" + "heuristic_" + df["heuristic"] + "_" + df[f"{heuristic}"]
    )

    blocks = get_blocks_information(df)

    iplot(plot_budget_utilization(blocks, total_tasks))
    iplot(plot_hard_run_ops(df, total_tasks))
    # tasks = get_tasks_information(df)
    # analyze_cache_type_use(experiment_path)


def analyze_cache_type_use(experiment_path):
    def plot_cache_type_use(df):
        fig = px.bar(
            df,
            x="id",
            y="count",
            color="type",
            title="",
            width=900,
            height=500,
            facet_row="HT",
        )
        return fig

    df = get_df(experiment_path)
    df["heuristic_threshold"] = df["heuristic_threshold"].astype(str)
    df["key"] = df["heuristic_threshold"]

    tasks = get_tasks_information(df)
    tasks_deterministic = tasks[["id", "deterministic_runs", "heuristic_threshold"]]
    tasks_deterministic.insert(0, "type", "DeterministicCache")
    tasks_deterministic = tasks_deterministic.rename(
        columns={"deterministic_runs": "count"}
    )
    tasks_probabilistic = tasks[["id", "probabilistic_runs", "heuristic_threshold"]]
    tasks_probabilistic.insert(0, "type", "ProbabilisticCache")
    tasks_probabilistic = tasks_probabilistic.rename(
        columns={"probabilistic_runs": "count"}
    )
    tasks = pd.concat([tasks_deterministic, tasks_probabilistic])
    tasks = tasks.rename(columns={"heuristic_threshold": "HT"})

    iplot(plot_cache_type_use(tasks))
