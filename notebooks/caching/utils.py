import pandas as pd
import json
from pathlib import Path
import plotly.express as px
from plotly.offline import iplot
from precycle.utils.utils import LOGS_PATH
from experiments.ray.analysis import load_ray_experiment

global_order = "64"
pure_dp = True


def get_df(path):
    logs = LOGS_PATH.joinpath("ray/" + path)
    df = load_ray_experiment(logs)
    return df


def get_tasks_information(df):
    dfs = []
    for (tasks, key, zipf_k) in df[["tasks_info", "key", "zipf_k"]].values:

        # total_pmw_runs = 0
        # total_laplace_runs = 0

        for task in tasks:
            if task["status"] == "finished":
                # laplace_runs = task["laplace_runs"]
                # total_laplace_runs += laplace_runs
                # pmw_runs = task["pmw_runs"]
                # total_pmw_runs += pmw_runs
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
                                # "total_laplace_runs": total_laplace_runs,
                                # "total_pmw_runs": total_pmw_runs,
                                # "laplace_runs": laplace_runs,
                                # "pmw_runs": pmw_runs,
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


def get_best_alphas(df):
    for (blocks, key) in df[["block_budgets_info", "key"]].values:

        max_orders = {}
        for block_id, budget in blocks:
            if global_order == None:
                orders = budget["orders"]
                order = max(orders, key=orders.get)
                max_orders[block_id] = order
            else:
                max_orders[block_id] = global_order
        print(key, max_orders)


def get_budgets_information(df, blocks):
    dfs = []
    global_dfs = []

    for (
        tasks,
        blocks,
        initial_budget,
        key,
        zipf_k,
        mechanism,
        learning_rate,
        warmup,
    ) in df[
        [
            "tasks_info",
            "block_budgets_info",
            "blocks_initial_budget",
            "key",
            "zipf_k",
            "mechanism",
            "learning_rate",
            "warmup",
        ]
    ].values:

        # for each block find the order with the max remaining capacity
        if not pure_dp:
            max_orders = {}
            for block_id, budget in blocks:
                if global_order == None:
                    orders = budget["orders"]
                    order = max(orders, key=orders.get)
                    max_orders[block_id] = order
                else:
                    max_orders[block_id] = global_order

        for i, task in enumerate(tasks):
            if not (i % 100 == 0 or i == len(tasks) - 1):
                continue

            run_metadata = task["run_metadata"]
            budget_per_block = run_metadata["budget_per_block"]
            for block, budget in budget_per_block.items():
                if not pure_dp:
                    task_budget = budget["orders"][max_orders[str(block)]]
                else:
                    task_budget = budget["epsilon"]

            global_budget = 0
            budget_per_block = run_metadata["accummulated_budget_per_block"]
            for block, budget in budget_per_block.items():
                if not pure_dp:
                    accummulated_budget = budget["orders"][max_orders[str(block)]]
                    global_budget += (
                        accummulated_budget
                        / initial_budget["orders"][max_orders[str(block)]]
                    )
                else:
                    accummulated_budget = budget["epsilon"]
                    global_budget += accummulated_budget

                dfs.append(
                    pd.DataFrame(
                        [
                            {
                                "id": task["id"],
                                "block": str(block),
                                "cumulative_budget": accummulated_budget,
                                "budget": task_budget,
                                "key": key,
                                "zipf_k": zipf_k,
                                "mechanism": mechanism,
                                "learning_rate": learning_rate,
                                "warmup": warmup,
                            }
                        ]
                    )
                )
            if not pure_dp:
                global_budget /= len(blocks)

            global_dfs.append(
                pd.DataFrame(
                    [
                        {
                            "id": task["id"],
                            "key": key,
                            "global_budget": global_budget,
                            "zipf_k": zipf_k,
                            "mechanism": mechanism,
                            "learning_rate": learning_rate,
                            "warmup": warmup,
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

    for (
        blocks,
        initial_budget,
        key,
        zipf_k,
        total_tasks,
        mechanism,
        learning_rate,
        warmup,
    ) in df[
        [
            "block_budgets_info",
            "blocks_initial_budget",
            "key",
            "zipf_k",
            "total_tasks",
            "mechanism",
            "learning_rate",
            "warmup",
        ]
    ].values:
        for block_id, budget in blocks:
            if not pure_dp:
                orders = budget["orders"]
                if global_order == None:
                    order = max(orders, key=orders.get)
                else:
                    order = global_order

                max_available_budget = orders[order]
                max_initial_budget = initial_budget["orders"][order]
            else:
                max_available_budget = budget["epsilon"]
                max_initial_budget = initial_budget

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
                            "mechanism": mechanism,
                            "learning_rate": learning_rate,
                            "warmup": warmup,
                        }
                    ]
                )
            )
    if dfs:
        blocks = pd.concat(dfs)
        return blocks
    return None


def get_sv_misses_information(df):
    dfs = []
    for (sv_misses, key, zipf_k) in df[["sv_misses", "key", "zipf_k"]].values:
        # print(sv_misses)
        for sv_node_id, misses in sv_misses.items():
            dfs.append(
                pd.DataFrame(
                    [
                        {
                            "sv_node_id": sv_node_id,
                            "misses": misses,
                            "key": key,
                            "zipf_k": zipf_k,
                        }
                    ]
                )
            )
    if dfs:
        dfs = pd.concat(dfs)
        return dfs
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
        fig.write_image(LOGS_PATH.joinpath(f"ray/{experiment_path}/num_allocated_tasks.png"))
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
        return fig

    def plot_budget_utilization(df, total_tasks):
        df["budget"] = df["initial_budget"] - df["budget"]
        # / df[
        # "initial_budget"
        # ]
        df.to_csv(LOGS_PATH.joinpath(f"ray/{experiment_path}/budget_utilization.csv"), index=False)
        fig = px.bar(
            df,
            x="zipf_k",
            y="budget",
            color="key",
            barmode="group",
            title=f"Budget Consumption - total tasks-{total_tasks}",
            # category_orders=category_orders,
            width=900,
            height=500,
        )
        fig.write_image(LOGS_PATH.joinpath(f"ray/{experiment_path}/budget_utilization.png"))
        return fig

    def plot_cumulative_budget_utilization_time(df, total_tasks):
        df = df.drop(columns="index")
        df.to_csv(LOGS_PATH.joinpath(f"ray/{experiment_path}/cumulative_budget_utilization.csv"), index=False)
        df["cumulative_budget"] = df["cumulative_budget"].astype(float)
        fig = px.line(
            df,
            x="id",
            y="cumulative_budget",
            color="key",
            title=f"Cumulative Budget Consumption - total tasks-{total_tasks}",
            # category_orders=category_orders,
            width=1000,
            height=600,
            facet_row="zipf_k",
        )
        fig.write_image(LOGS_PATH.joinpath(f"ray/{experiment_path}/cumulative_budget_utilization.png"))
        return fig

    def plot_budget_utilization_time(df, total_tasks):
        fig = px.scatter(
            df,
            x="id",
            y="budget",
            color="key",
            title=f"Budget Consumption per task/time - total tasks-{total_tasks}",
            width=2500,
            height=800,
            facet_row="zipf_k",
        )
        return fig

    df = get_df(experiment_path)
    df["zipf_k"] = df["zipf_k"].astype(float)
    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)
    total_tasks = df["total_tasks"].max()
    blocks = get_blocks_information(df)

    iplot(plot_num_allocated_tasks(df, total_tasks))
    iplot(plot_total_sv_checks(df, total_tasks))
    iplot(plot_total_sv_misses(df, total_tasks))
    iplot(plot_budget_utilization(blocks, total_tasks))
    time_budgets, _ = get_budgets_information(df, df["block_budgets_info"])
    iplot(plot_cumulative_budget_utilization_time(time_budgets, total_tasks))
    iplot(plot_budget_utilization_time(time_budgets, total_tasks))

    # analyze_mechanism_type_use(df)
    # analyze_mechanism_type_use_bar(df)


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
        df["budget"] = df["initial_budget"] - df["budget"]
        # / df[
        #     "initial_budget"
        # ]
        keys_order = [
            "Hybrid:bin_visits:100-10:lr0.2:bsFalse",
            "DirectLaplace",
            "PMW",
        ]
        category_orders = {"key": keys_order}

        df["block"] = df["id"]
        fig = px.bar(
            df,
            x="block",
            y="budget",
            color="key",
            barmode="group",
            title=f"Absolute Budget Consumption - total tasks-{total_tasks}",
            width=1200,
            height=600,
            category_orders=category_orders,
            facet_row="zipf_k",
        )
        return fig

    def plot_total_sv_checks(df, total_tasks):
        keys_order = [
            "Hybrid:bin_visits:100-10:lr0.2:bsFalse",
            "DirectLaplace",
            "PMW",
        ]
        category_orders = {"key": keys_order}
        fig = px.bar(
            df,
            x="zipf_k",
            y="total_sv_checks",
            color="key",
            barmode="group",
            title=f"Total SV checks - total tasks={total_tasks}",
            category_orders=category_orders,
            width=900,
            height=500,
        )
        return fig

    def plot_total_sv_misses(df, total_tasks):
        keys_order = [
            "Hybrid:bin_visits:100-10:lr0.2:bsFalse",
            "DirectLaplace",
            "PMW",
        ]
        category_orders = {"key": keys_order}
        fig = px.bar(
            df,
            x="zipf_k",
            y="total_sv_misses",
            color="key",
            barmode="group",
            title=f"Total SV failures - total tasks={total_tasks}",
            category_orders=category_orders,
            width=900,
            height=500,
        )
        return fig

    def plot_global_budget_utilization_time(df, total_tasks):
        df["task"] = df["id"]
        df["cumul_budget"] = df["global_budget"]
        keys_order = [
            "Hybrid:bin_visits:100-10:lr0.2:bsFalse",
            "DirectLaplace",
            "PMW",
        ]
        category_orders = {"key": keys_order}
        fig = px.line(
            df,
            x="task",
            y="cumul_budget",
            color="key",
            title=f"Cumulative Absolute Budget Consumption across all blocks - total tasks-{total_tasks}",
            width=1000,
            height=600,
            facet_row="zipf_k",
            category_orders=category_orders,
        )
        return fig

    # def plot_budget_utilization_time_accum(df, total_tasks):
    #     fig = px.line(
    #         df,
    #         x="id",
    #         y="cumulative_budget",
    #         color="key",
    #         title=f"Budget Utilization per task/time - total tasks-{total_tasks}",
    #         width=2500,
    #         height=800,
    #         facet_row="zipf_k",
    #     )
    #     return fig

    # def plot_budget_utilization_time(df, total_tasks):
    #     fig = px.scatter(
    #         df,
    #         x="id",
    #         y="budget",
    #         color="key",
    #         title=f"Budget Utilization per task/time - total tasks-{total_tasks}",
    #         width=2500,
    #         height=800,
    #         facet_row="zipf_k",
    #         range_y=[0, 1],
    #     )
    #     return fig

    df = get_df(experiment_path)
    total_tasks = df["total_tasks"].max()
    df["zipf_k"] = df["zipf_k"].astype(float)
    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)
    iplot(plot_num_allocated_tasks(df, total_tasks))
    blocks = get_blocks_information(df)
    iplot(plot_budget_utilization(blocks, total_tasks))
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


def analyze_sv_misses(experiment_path):
    def plot_sv_misses_per_node(df, total_tasks):
        df["sparse_vector_node"] = df["sv_node_id"]
        df["number_of_misses"] = df["misses"]

        fig = px.bar(
            df,
            x="sparse_vector_node",
            y="number_of_misses",
            color="key",
            barmode="group",
            title=f"Number of misses per Sparse Vector - total tasks={total_tasks}",
            width=900,
            height=500,
            facet_row="zipf_k",
        )
        return fig

    df = get_df(experiment_path)
    df = df.query("mechanism == 'Hybrid'")
    total_tasks = df["total_tasks"].max()
    sv_misses = get_sv_misses_information(df)
    iplot(plot_sv_misses_per_node(sv_misses, total_tasks))


def analyze_num_cuts(experiment_path):
    def plot_num_cuts(df):
        df["node_size"] = df["chunk_size"]
        df["node"] = df["chunk"]

        fig = px.histogram(df, x="node", color="node_size", width=750, height=400)
        return fig

    df = get_df(experiment_path)
    df = df.query("zipf_k == 0.5")

    chunks_list = []
    chunk_sizes = []
    chunks = df["chunks"][0]
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


def analyze_query_types(experiment_path):
    def plot_query_per_task(df):

        # fig = px.histogram(
        #     df,
        #     x="query_id",
        #     title="Query Pool Histograms",
        #     # color="query_id",
        #     width=800,
        #     height=500,
        #     facet_row="zipf_k",
        # )
        # return fig

        fig = px.bar(
            df,
            x="zipf_k",
            y="count",
            title="Num Different queries",
            # color="query_id",
            width=800,
            height=500,
            # facet_row="zipf_k",
        )
        return fig

    df = get_df(experiment_path)
    df["zipf_k"] = df["zipf_k"].astype(float)
    df = df.query("mechanism == 'DirectLaplace'")
    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)

    tasks = get_tasks_information(df)
    tasks = tasks[["id", "key", "query_id", "zipf_k"]]
    print(tasks.groupby("zipf_k")["query_id"].nunique())
    # iplot(plot_query_per_task(tasks))


# def analyze_mechanism_type_use_bar(df):
#     def plot_mechanism_type_use(df):
#         fig = px.bar(
#             df,
#             x="id",
#             y="count",
#             color="type",
#             title="Laplace vs PMW runs for mechanism",
#             width=3500,
#             height=800,
#             facet_row="zipf_k",
#             color_discrete_sequence=["red", "black"],
#             # color_discrete_map={
#             #     'Deterministicmechanism': 'red',
#             #     'Probabilisticmechanism': 'black'
#             # }
#         )
#         return fig

#     df = df.query("mechanism == 'MixedRuns'")
#     tasks = get_tasks_information(df)
#     tasks_deterministic = tasks[["id", "laplace_runs", "zipf_k"]]
#     tasks_deterministic.insert(0, "type", "LaplaceRuns")
#     tasks_deterministic = tasks_deterministic.rename(columns={"laplace_runs": "count"})
#     tasks_probabilistic = tasks[["id", "pmw_runs", "zipf_k"]]
#     tasks_probabilistic.insert(0, "type", "PMWRuns")
#     tasks_probabilistic = tasks_probabilistic.rename(columns={"pmw_runs": "count"})
#     tasks_new = pd.concat([tasks_deterministic, tasks_probabilistic])
#     iplot(plot_mechanism_type_use(tasks_new))


# def analyze_mechanism_type_use(df):
#     def plot_mechanism_type_use(df):
#         fig = px.line(
#             df,
#             x="id",
#             y="total_pmw_runs",
#             color="zipf_k",
#             title="PMW runs",
#             width=3500,
#             height=800,
#         )
#         return fig

#     df = df.query("mechanism == 'MixedRuns'")
#     tasks = get_tasks_information(df)
#     iplot(plot_mechanism_type_use(tasks))
