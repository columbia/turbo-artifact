import pandas as pd
import json

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

        total_pmw_runs = 0
        total_laplace_runs = 0

        for task in tasks:
            if task["status"] == "finished":
                laplace_runs = task["laplace_runs"]
                total_laplace_runs += laplace_runs
                pmw_runs = task["pmw_runs"]
                total_pmw_runs += pmw_runs
                computations_aggregated = len(task["run_metadata"].keys())

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
                                "computations_aggregated": computations_aggregated,
                                "key": key,
                                "zipf_k": zipf_k,
                                "total_laplace_runs": total_laplace_runs,
                                "total_pmw_runs": total_pmw_runs,
                                "laplace_runs": laplace_runs,
                                "pmw_runs": pmw_runs,
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
    global_dfs = []
    # gbudget_dfs = []

    for (tasks, blocks, initial_budget, key, zipf_k) in df[
        ["tasks_info", "block_budgets_info", "blocks_initial_budget", "key", "zipf_k"]
    ].values:

        # for each block find the order with the max remaining capacity
        max_orders = {}
        for block_id, budget in blocks:
            orders = budget["orders"]
            order = max(orders, key=orders.get)
            max_orders[block_id] = order

        for task in tasks:

            # if task["id"] % 500 != 0:
            #     continue

            accummulated_budget_per_block = {}
            global_budget = 0

            run_metadata = task["run_metadata"]
            if run_metadata is None:
                continue
            budget_per_block = run_metadata["budget_per_block"]
            for block, budget in budget_per_block.items():
                task_budget = budget["orders"][max_orders[str(block)]]

            budget_per_block = run_metadata["accummulated_budget_per_block"]
            for block, budget in budget_per_block.items():
                budget = budget["orders"][max_orders[str(block)]]
                accummulated_budget_per_block[str(block)] = budget
                global_budget += budget

                dfs.append(
                    pd.DataFrame(
                        [
                            {
                                "id": task["id"],
                                "block": str(block),
                                "accummulated_budget": accummulated_budget_per_block[
                                    str(block)
                                ],
                                "budget": task_budget,
                                "key": key,
                                "zipf_k": zipf_k,
                            }
                        ]
                    )
                )
            global_dfs.append(
                pd.DataFrame(
                    [
                        {
                            "id": task["id"],
                            "key": key,
                            "global_budget": global_budget,
                            "zipf_k": zipf_k,
                        }
                    ]
                )
            )

    if dfs:
        time_budgets = pd.concat(dfs).reset_index()
        global_time_budgets = pd.concat(global_dfs).reset_index()
        return time_budgets, global_time_budgets
    return None


def get_blocks_information(df):
    dfs = []

    for (blocks, initial_budget, key, zipf_k, total_tasks) in df[
        ["block_budgets_info", "blocks_initial_budget", "key", "zipf_k", "total_tasks"]
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
                            "total_tasks": total_tasks,
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

    def plot_total_sv_checks(df, total_tasks):

        fig = px.bar(
            df,
            x="zipf_k",
            y="total_sv_checks",
            color="key",
            barmode="group",
            title=f"Total SV checks - total tasks={total_tasks}",
            width=900,
            height=500,
        )
        return fig

    def plot_total_sv_misses(df, total_tasks):

        fig = px.bar(
            df,
            x="zipf_k",
            y="total_sv_misses",
            color="key",
            barmode="group",
            title=f"Total SV failures - total tasks={total_tasks}",
            width=900,
            height=500,
        )
        # fig.add_bar(
        #     x=df['zipf_k'],
        #     y=df['total_sv_misses'],
        #     # barmode="group",
        #     # width=900,
        #     # height=500,
        #     )
        return fig

    def plot_budget_utilization(df, total_tasks):
        df["budget_consumption"] = df["initial_budget"] - df["budget"]
        # / df[
        # "initial_budget"
        # ]
        keys_order = ["MixedRuns", "LaplaceRuns", "PMWRuns"]
        zipf_orders = ["1.5", "1.0", "0.5", "0"]
        category_orders = {"key": keys_order, "zipf_k": zipf_orders}

        fig = px.bar(
            df,
            x="zipf_k",
            y="budget_consumption",
            color="key",
            barmode="group",
            title=f"Budget Consumption - total tasks-{total_tasks}",
            width=900,
            height=500,
            category_orders=category_orders,
            color_discrete_map={
                "DeterministicCache": "blue",
                "ProbabilisticCache": "green",
            },
            # range_y=[0, 1],
        )
        return fig

    def plot_cumulative_budget_utilization_time(df, total_tasks):
        keys_order = ["MixedRuns", "LaplaceRuns", "PMWRuns"]
        category_orders = {"key": keys_order}

        fig = px.line(
            df,
            x="id",
            y="accummulated_budget",
            color="key",
            title=f"Cumulative Budget Consumption per task/time - total tasks-{total_tasks}",
            width=2500,
            height=800,
            category_orders=category_orders,
            facet_row="zipf_k",
            # range_y=[0, 1],
        )

        # fig.write_html("plot.html")
        return fig

    def plot_budget_utilization_time(df, total_tasks):
        keys_order = ["MixedRuns", "LaplaceRuns", "PMWRuns"]
        category_orders = {"key": keys_order}

        fig = px.scatter(
            df,
            x="id",
            y="budget",
            color="key",
            title=f"Budget Consumption per task/time - total tasks-{total_tasks}",
            width=2500,
            height=800,
            category_orders=category_orders,
            facet_row="zipf_k",
            # range_y=[0, 1],
        )

        # fig.write_html("plot.html")
        return fig

    df = get_df(experiment_path)
    # print(df['total_sv_misses'])
    df["heuristic"] = df["heuristic"].astype(str)
    df["learning_rate"] = df["learning_rate"].astype(str)
    df["key"] = df["cache"] + ":" + df["heuristic"] + ":" + df["learning_rate"]
    df["zipf_k"] = df["zipf_k"].astype(float)
    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)

    total_tasks = df["total_tasks"].max()
    blocks = get_blocks_information(df)

    iplot(plot_num_allocated_tasks(df, total_tasks))
    iplot(plot_total_sv_checks(df, total_tasks))
    iplot(plot_total_sv_misses(df, total_tasks))
    iplot(plot_budget_utilization(blocks, total_tasks))
    time_budgets, global_budgets = get_budgets_information(df, df["block_budgets_info"])
    iplot(plot_cumulative_budget_utilization_time(time_budgets, total_tasks))
    iplot(plot_budget_utilization_time(time_budgets, total_tasks))

    # analyze_cache_type_use(df)
    # analyze_cache_type_use_bar(df)


def plot_budget_utilization_total_tasks(experiment_path):
    def plot_budget(df):
        df["budget_utilization"] = (df["initial_budget"] - df["budget"]) / df[
            "initial_budget"
        ]
        keys_order = ["MixedRuns", "LaplaceRuns", "PMWRuns"]
        total_tasks_orders = ["10000", "50000", "100000"]
        category_orders = {"key": keys_order, "zipf_k": total_tasks_orders}

        fig = px.bar(
            df,
            x="total_tasks",
            y="budget_utilization",
            color="key",
            barmode="group",
            title=f"Budget Utilization total tasks x axis",
            width=900,
            height=500,
            category_orders=category_orders,
            range_y=[0, 1],
        )
        return fig

    df = get_df(experiment_path)
    df["heuristic"] = df["heuristic"].astype(str)
    df["total_tasks"] = df["total_tasks"].astype(str)
    df["key"] = df["cache"] + df["heuristic"]
    df["zipf_k"] = df["zipf_k"].astype(float)
    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)
    blocks = get_blocks_information(df)
    iplot(plot_budget(blocks))


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


def analyze_cache_type_use_bar(df):
    def plot_cache_type_use(df):
        fig = px.bar(
            df,
            x="id",
            y="count",
            color="type",
            title="Laplace vs PMW runs for cache",
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

    df = df.query("cache == 'MixedRuns'")
    tasks = get_tasks_information(df)
    tasks_deterministic = tasks[["id", "laplace_runs", "zipf_k"]]
    tasks_deterministic.insert(0, "type", "LaplaceRuns")
    tasks_deterministic = tasks_deterministic.rename(columns={"laplace_runs": "count"})
    tasks_probabilistic = tasks[["id", "pmw_runs", "zipf_k"]]
    tasks_probabilistic.insert(0, "type", "PMWRuns")
    tasks_probabilistic = tasks_probabilistic.rename(columns={"pmw_runs": "count"})
    tasks_new = pd.concat([tasks_deterministic, tasks_probabilistic])
    iplot(plot_cache_type_use(tasks_new))


def analyze_cache_type_use(df):
    def plot_cache_type_use(df):
        fig = px.line(
            df,
            x="id",
            y="total_pmw_runs",
            color="zipf_k",
            title="PMW runs",
            width=3500,
            height=800,
        )
        return fig

    df = df.query("cache == 'MixedRuns'")
    tasks = get_tasks_information(df)
    iplot(plot_cache_type_use(tasks))


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
        # keys_order = ["CombinedCache", "ProbabilisticCache", "CombinedCache"]
        # category_orders = {"key": keys_order}

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
            # category_orders=category_orders,
            # range_y=[0, 1],
        )
        return fig

    def plot_num_cuts(df, total_tasks):
        # keys_order = ["CombinedCache", "ProbabilisticCache", "CombinedCache"]
        # zipf_orders = ["1.5", "1.0", "0.5", "0"]
        # category_orders = {"key": keys_order}

        fig = px.histogram(
            df,
            x="computations_aggregated",
            color="key",
            barmode="group",
            title=f"Computations aggregated - total tasks={total_tasks}",
            width=2500,
            height=800,
            # category_orders=category_orders,
            # color_discrete_map={
            # "DeterministicCache": "blue",
            # "ProbabilisticCache": "green",
            # },
            range_y=[0, 50],
            facet_row="zipf_k",
        )
        return fig

    # df = get_df(experiment_path)
    logs = LOGS_PATH.joinpath("ray/" + experiment_path)
    results = []
    for run_result in logs.glob("**/result.json"):
        try:
            with open(run_result, "r") as f:
                d = json.load(f)
                del d["tasks_info"]
            results.append(d)
        except Exception:
            pass
    df = pd.DataFrame(results)
    total_tasks = df["total_tasks"].max()
    df["learning_rate"] = df["learning_rate"].astype(str)
    df["heuristic"] = df["heuristic"].astype(str)
    df["bootstrapping"] = df["bootstrapping"].astype(str)
    df["key"] = df["cache"] + ":" + df["heuristic"] + ":" + df["learning_rate"] + ":" + df["bootstrapping"]
    df["zipf_k"] = df["zipf_k"].astype(float)
    iplot(plot_num_allocated_tasks(df, total_tasks))
    blocks = get_blocks_information(df)
    iplot(plot_budget_utilization(blocks, total_tasks))


def budget_utilization_time(experiment_path):
    def plot_total_sv_checks(df, total_tasks):

        fig = px.bar(
            df,
            x="zipf_k",
            y="total_sv_checks",
            color="key",
            barmode="group",
            title=f"Total SV checks - total tasks={total_tasks}",
            width=900,
            height=500,
        )
        return fig

    def plot_total_sv_misses(df, total_tasks):

        fig = px.bar(
            df,
            x="zipf_k",
            y="total_sv_misses",
            color="key",
            barmode="group",
            title=f"Total SV failures - total tasks={total_tasks}",
            width=900,
            height=500,
        )
        return fig

    def plot_global_budget_utilization_time(df, total_tasks):
        # keys_order = ["MixedRuns", "LaplaceRuns", "PMWRuns"]
        # category_orders = {"key": keys_order}
        fig = px.line(
            df,
            x="id",
            y="global_budget",
            color="key",
            title=f"Global Budget Utilization per task/time - total tasks-{total_tasks}",
            width=2500,
            height=800,
            # category_orders=category_orders,
            facet_row="zipf_k",
            # range_y=[0, 1],
        )
        # fig.write_html("plot.html")
        return fig

    def plot_budget_utilization_time_accum(df, total_tasks):
        fig = px.line(
            df,
            x="id",
            y="accummulated_budget",
            color="key",
            title=f"Budget Utilization per task/time - total tasks-{total_tasks}",
            width=2500,
            height=800,
            facet_row="zipf_k",
        )
        return fig

    # def plot_budget_utilization_time(df, total_tasks):
    #     # keys_order = ["MixedRuns", "LaplaceRuns", "PMWRuns"]
    #     # category_orders = {"key": keys_order}

    #     fig = px.scatter(
    #         df,
    #         x="id",
    #         y="budget",
    #         color="key",
    #         title=f"Budget Utilization per task/time - total tasks-{total_tasks}",
    #         width=2500,
    #         height=800,
    #         # category_orders=category_orders,
    #         facet_row="zipf_k",
    #         range_y=[0, 1],
    #     )

    #     fig.write_html("plot.html")
    #     return fig

    df = get_df(experiment_path)
    total_tasks = df["total_tasks"].max()
    df["heuristic"] = df["heuristic"].astype(str)
    df["learning_rate"] = df["learning_rate"].astype(str)
    df["bootstrapping"] = df["bootstrapping"].astype(str)
    df["key"] = df["cache"] + ":" + df["heuristic"] + ":" + df["learning_rate"] + ":" + df["bootstrapping"]

    # df["key"] = df["cache"] + ":" + df["heuristic"] + ":" + df["learning_rate"]
    df["zipf_k"] = df["zipf_k"].astype(float)

    iplot(plot_total_sv_checks(df, total_tasks))
    iplot(plot_total_sv_misses(df, total_tasks))

    time_budgets, global_time_budgets = get_budgets_information(
        df, df["block_budgets_info"]
    )
    # # print(time_budgets)
    iplot(plot_global_budget_utilization_time(global_time_budgets, total_tasks))

    # # Plot for block 0
    # d = time_budgets.query("block == '0'")
    # iplot(plot_budget_utilization_time_accum(d, total_tasks))
    # # iplot(plot_budget_utilization_time(d, total_tasks))

    # # Plot for block 1
    # d = time_budgets.query("block == '1'")
    # iplot(plot_budget_utilization_time_accum(d, total_tasks))
    # # iplot(plot_budget_utilization_time(d, total_tasks))

    # # Plot for block 2
    # d = time_budgets.query("block == '2'")
    # iplot(plot_budget_utilization_time_accum(d, total_tasks))
    # # iplot(plot_budget_utilization_time(d, total_tasks))

    # # Plot for block 3
    # d = time_budgets.query("block == '3'")
    # iplot(plot_budget_utilization_time_accum(d, total_tasks))
    # # iplot(plot_budget_utilization_time(d, total_tasks))

    # # Plot for block 4
    # d = time_budgets.query("block == '4'")
    # iplot(plot_budget_utilization_time_accum(d, total_tasks))
    # # iplot(plot_budget_utilization_time(d, total_tasks))


def analyze_num_cuts(experiment_path):
    def plot_num_cuts(df):
        # keys_order = ["CombinedCache", "ProbabilisticCache", "CombinedCache"]
        # zipf_orders = ["1.5", "1.0", "0.5", "0"]
        # category_orders = {"key": keys_order}

        fig = px.histogram(df, x="chunk", color="chunk_size", width=750, height=400)
        return fig

    df = get_df(experiment_path)
    df["heuristic"] = df["heuristic"].astype(str)
    df = df.query("cache == 'MixedCache' and zipf_k == 0.5")
    df["key"] = df["cache"] + ":" + df["heuristic"]

    # df["zipf_k"] = df["zipf_k"].astype(float)
    # df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)

    chunks_list = []
    chunk_sizes = []
    chunks = df['chunks'][0]
    for chunk, occurences in chunks.items():
        ch = chunk[1:-1]
        b1, b2 = ch.split(",")
        for _ in range(occurences):
            chunks_list.append(ch)
            chunk_sizes.append(int(b2) - int(b1) + 1)

    df1 = pd.DataFrame(chunks_list, columns=["chunk"])
    df2 = pd.DataFrame(chunk_sizes, columns=["chunk_size"])
    df = pd.concat([df1, df2], axis=1)
    df.sort_values(["chunk_size"], ascending=[False], inplace=True)

    # print(df)
    iplot(plot_num_cuts(df))
    # tasks = get_tasks_information(df)
    # iplot(plot_num_cuts(tasks, total_tasks))
