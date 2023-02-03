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
    for (tasks, key, zipf_k) in df[["tasks_info", "key", "zipf_k"]].values:

        for task in tasks:

            deterministic_runs = task["deterministic_runs"]
            probabilistic_runs = task["probabilistic_runs"]

            dfs.append(
                pd.DataFrame(
                    [
                        {
                            "id": task["id"],
                            "result": task["result"],
                            "query_id": task["query_id"],
                            # "error": task["error"],
                            "planning_time": task["planning_time"],
                            "blocks": task["blocks"],
                            "n_blocks": task["n_blocks"],
                            # "hard_run_ops": task["run_metadata"]["hard_run_ops"],
                            "utility": task["utility"],
                            "utility_beta": task["utility_beta"],
                            "status": task["status"],
                            "key": key,
                            "zipf_k": zipf_k,
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


def get_budgets_information(df, blocks):
    dfs = []

    for (tasks, blocks, initial_budget, key, zipf_k) in df[
        ["tasks_info", "block_budgets_info", "blocks_initial_budget", "key", "zipf_k"]
    ].values:
        total_budget_per_block = {}

        max_orders = {}
        for block_id, budget in blocks:
            orders = budget["orders"]
            order = max(orders, key=orders.get)
            max_orders[block_id] = order
            total_budget_per_block[block_id] = 0

        for task in tasks:
            run_metadata = task["run_metadata"]
            for run_op, run_op_metadata in run_metadata.items():
                s = run_op.split(",")
                l = int(s[0].split("(")[1])
                r = int(s[1].split(")")[0])
                orders = run_op_metadata["run_budget"]["orders"]
                for bid in range(l, r + 1):
                    budget = orders[max_orders[str(bid)]]
                    max_initial_budget = initial_budget["orders"][max_orders[str(bid)]]
                    total_budget_per_block[str(bid)] += budget
                    dfs.append(
                        pd.DataFrame(
                            [
                                {
                                    "id": task["id"],
                                    "block": bid,
                                    "budget": total_budget_per_block[str(bid)]
                                    / max_initial_budget,
                                    "key": key,
                                    "zipf_k": zipf_k,
                                }
                            ]
                        )
                    )

    if dfs:
        time_budgets = pd.concat(dfs)
        return time_budgets
    return None


def get_blocks_information(df):
    dfs = []

    for (blocks, initial_budget, key, zipf_k) in df[
        [
            "block_budgets_info",
            "blocks_initial_budget",
            "key",
            "zipf_k",
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
                            "zipf_k": zipf_k,
                        }
                    ]
                )
            )
    if dfs:
        blocks = pd.concat(dfs)
        return blocks
    return None


def analyze_monoblock(experiment_path):
    def plot_num_allocated_tasks(df, total_tasks):

        fig = px.bar(
            df,
            x="zipf_k",
            y="n_allocated_tasks",
            color="key",
            barmode="group",
            title=f"Num allocated tasks - total tasks={total_tasks}",
            width=900,
            height=500,
        )
        return fig

    def plot_budget_utilization(df, total_tasks):
        df["budget_utilization"] = (df["initial_budget"] - df["budget"]) / df[
            "initial_budget"
        ]
        keys_order = ["CombinedCache", "ProbabilisticCache", "CombinedCache"]
        zipf_orders = ["1.5", "1.0", "0.5", "0"]
        category_orders = {"key": keys_order, "zipf_k": zipf_orders}

        fig = px.bar(
            df,
            x="zipf_k",
            y="budget_utilization",
            color="key",
            barmode="group",
            title=f"Budget Utilization - total tasks-{total_tasks}",
            width=900,
            height=500,
            category_orders=category_orders,
            color_discrete_map={
                "DeterministicCache": "blue",
                "ProbabilisticCache": "green",
            },
            range_y=[0, 1],
        )
        return fig

    def plot_hard_run_ops(df, total_tasks):
        keys_order = ["CombinedCache", "ProbabilisticCache", "CombinedCache"]
        zipf_orders = ["1.5", "1.0", "0.5", "0"]
        category_orders = {"key": keys_order, "zipf_k": zipf_orders}

        fig = px.bar(
            df,
            x="zipf_k",
            y="avg_total_hard_run_ops",
            color="key",
            barmode="group",
            title=f"Hard Queries - total tasks={total_tasks}",
            width=900,
            height=500,
            category_orders=category_orders,
            # color_discrete_map={
            # "DeterministicCache": "blue",
            # "ProbabilisticCache": "green",
            # },
            range_y=[0, 1],
        )
        return fig

    def plot_budget_utilization_time(df, total_tasks):
        keys_order = ["CombinedCache", "ProbabilisticCache", "CombinedCache"]
        category_orders = {"key": keys_order}

        fig = px.line(
            df,
            x="id",
            y="budget",
            color="key",
            title=f"Budget Utilization per task/time - total tasks-{total_tasks}",
            width=2500,
            height=800,
            category_orders=category_orders,
            facet_row="zipf_k",
            range_y=[0, 1],
        )

        fig.write_html("plot.html")
        return fig

    df = get_df(experiment_path)
    df["key"] = df["cache"]
    df["zipf_k"] = df["zipf_k"].astype(float)
    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)

    total_tasks = df["total_tasks"].max()
    blocks = get_blocks_information(df)
    # df = df.query("(cache == 'CombinedCache' or cache == 'DeterministicCache') and zipf_k == 0.0")
    time_budgets = get_budgets_information(df, df["block_budgets_info"]).reset_index()

    iplot(plot_num_allocated_tasks(df, total_tasks))
    iplot(plot_budget_utilization(blocks, total_tasks))
    iplot(plot_hard_run_ops(df, total_tasks))
    iplot(plot_budget_utilization_time(time_budgets, total_tasks))
    analyze_cache_type_use(df)


def analyze_query_types(experiment_path):
    def plot_query_per_task(df):
        fig = px.scatter(
            df,
            x="id",
            y="query_id",
            title="Query per task",
            color="key",
            width=3500,
            height=800,
            facet_row="zipf_k",
            color_discrete_sequence=["red", "black"],
            # color_discrete_map={
            #     'DeterministicCache': 'red',
            #     'ProbabilisticCache': 'black'
            # }
        )
        return fig

    df = get_df(experiment_path)
    df["key"] = df["cache"]
    df["zipf_k"] = df["zipf_k"].astype(float)
    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)
    tasks = get_tasks_information(df)
    tasks = tasks[["id", "key", "query_id", "zipf_k"]]
    iplot(plot_query_per_task(tasks))


def analyze_cache_type_use(df):
    def plot_cache_type_use(df):
        fig = px.bar(
            df,
            x="id",
            y="count",
            color="type",
            title="Deterministic vs Probabilistic runs for Combined cache",
            width=3500,
            height=800,
            facet_row="zipf_k",
            color_discrete_sequence=["red", "black"],
            # color_discrete_map={
            #     'DeterministicCache': 'red',
            #     'ProbabilisticCache': 'black'
            # }
        )
        return fig

    df = df.query("cache == 'CombinedCache'")
    tasks = get_tasks_information(df)
    tasks_deterministic = tasks[["id", "deterministic_runs", "zipf_k"]]
    tasks_deterministic.insert(0, "type", "DeterministicCache")
    tasks_deterministic = tasks_deterministic.rename(
        columns={"deterministic_runs": "count"}
    )
    tasks_probabilistic = tasks[["id", "probabilistic_runs", "zipf_k"]]
    tasks_probabilistic.insert(0, "type", "ProbabilisticCache")
    tasks_probabilistic = tasks_probabilistic.rename(
        columns={"probabilistic_runs": "count"}
    )
    tasks_new = pd.concat([tasks_deterministic, tasks_probabilistic])
    iplot(plot_cache_type_use(tasks_new))


def analyze_multiblock(experiment_path):
    def plot_num_allocated_tasks(df, total_tasks):

        fig = px.bar(
            df,
            x="zipf_k",
            y="n_allocated_tasks",
            color="key",
            barmode="group",
            title=f"Num allocated tasks - total tasks={total_tasks}",
            width=900,
            height=500,
        )
        return fig

    def plot_budget_utilization(df, total_tasks):
        df["budget_utilization"] = (df["initial_budget"] - df["budget"]) / df[
            "initial_budget"
        ]
        keys_order = ["CombinedCache", "ProbabilisticCache", "CombinedCache"]
        category_orders = {"key": keys_order}

        fig = px.bar(
            df,
            x="id",
            y="budget_utilization",
            color="key",
            barmode="group",
            title=f"Budget Utilization - total tasks-{total_tasks}",
            width=2500,
            height=800,
            facet_row="zipf_k",
            category_orders=category_orders,
            range_y=[0, 1],
        )
        return fig

    def plot_hard_run_ops(df, total_tasks):
        keys_order = ["CombinedCache", "ProbabilisticCache", "CombinedCache"]
        zipf_orders = ["1.5", "1.0", "0.5", "0"]
        category_orders = {"key": keys_order, "zipf_k": zipf_orders}

        fig = px.bar(
            df,
            x="zipf_k",
            y="avg_total_hard_run_ops",
            color="key",
            barmode="group",
            title=f"Hard Queries - total tasks={total_tasks}",
            width=900,
            height=500,
            category_orders=category_orders,
            # color_discrete_map={
            # "DeterministicCache": "blue",
            # "ProbabilisticCache": "green",
            # },
            range_y=[0, 1],
        )
        return fig

    # def plot_budget_utilization_time(df, total_tasks):
    #     keys_order = ["CombinedCache", "ProbabilisticCache", "CombinedCache"]
    #     # zipf_orders = ["1.5", "1.0", "0.5", "0"]
    #     category_orders = {"key": keys_order}

    #     fig = px.line(
    #         df,
    #         x="id",
    #         y="budget",
    #         color="key",
    #         title=f"Budget Utilization per task/time - total tasks-{total_tasks}",
    #         width=2500,
    #         height=800,
    #         category_orders=category_orders,
    #         facet_row="zipf_k",
    #         # range_y=[0, 1],
    #     )

    #     fig.write_html("plot.html")
    #     return fig

    df = get_df(experiment_path)
    df["key"] = df["cache"]
    df["zipf_k"] = df["zipf_k"].astype(float)
    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)

    total_tasks = df["total_tasks"].max()
    blocks = get_blocks_information(df)
    # time_budgets = get_budgets_information(df, df["block_budgets_info"]).reset_index()
    iplot(plot_num_allocated_tasks(df, total_tasks))
    iplot(plot_budget_utilization(blocks, total_tasks))
    # iplot(plot_hard_run_ops(df, total_tasks))
    # iplot(plot_budget_utilization_time(time_budgets, total_tasks))
