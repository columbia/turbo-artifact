import pandas as pd
import plotly.express as px
from plotly.offline import iplot
from precycle.utils.utils import LOGS_PATH
from experiments.ray.analysis import load_ray_experiment
import numpy as np


def get_df(path):
    logs = LOGS_PATH.joinpath(path)
    df = load_ray_experiment(logs)
    return df


def get_budgets_information(df):
    dfs = []
    global_dfs = []

    for (
        tasks,
        key,
        zipf_k,
        mechanism,
        learning_rate,
        warmup,
        planner,
        heuristic,
    ) in df[
        [
            "tasks_info",
            "key",
            "zipf_k",
            "mechanism",
            "learning_rate",
            "warmup",
            "planner",
            "heuristic",
        ]
    ].values:

        for i, task in enumerate(tasks):
            # if not (i % 100 == 0 or i == len(tasks) - 1):
            # continue

            run_metadata = task["run_metadata"]
            task_budget_per_block = {}
            budget_per_block = run_metadata["budget_per_block"]
            for block, budget in budget_per_block.items():
                task_budget_per_block[block] = budget["epsilon"]

            sum_budget_across_blocks = 0
            cumulative_budget_per_block = run_metadata["cumulative_budget_per_block"]
            for block, budget in cumulative_budget_per_block.items():
                cumulative_budget = budget["epsilon"]
                sum_budget_across_blocks += cumulative_budget
                task_block_budget = (
                    task_budget_per_block[block]
                    if block in task_budget_per_block
                    else 0
                )
                dfs.append(
                    pd.DataFrame(
                        [
                            {
                                "task": task["id"],
                                "cumulative_budget": cumulative_budget,
                                "mechanism": mechanism,
                                "zipf_k": zipf_k,
                                "block": str(block),
                                "budget": task_block_budget,
                                "key": key,
                                "learning_rate": learning_rate,
                                "warmup": warmup,
                                "planner": planner,
                                "heuristic": heuristic,
                            }
                        ]
                    )
                )

            global_dfs.append(
                pd.DataFrame(
                    [
                        {
                            "task": task["id"],
                            "sum_budget_across_blocks": sum_budget_across_blocks,
                            "mechanism": mechanism,
                            "zipf_k": zipf_k,
                            "key": key,
                            "learning_rate": learning_rate,
                            "warmup": warmup,
                            "planner": planner,
                            "heuristic": heuristic,
                        }
                    ]
                )
            )

    if dfs:
        time_budgets = pd.concat(dfs).reset_index()
        time_budgets["cumulative_budget"] = time_budgets["cumulative_budget"].astype(
            float
        )
        time_budgets = time_budgets.drop(columns="index")
        global_time_budgets = pd.concat(global_dfs).reset_index()
        global_time_budgets = global_time_budgets.drop(columns="index")
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
        planner,
        heuristic,
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
            "planner",
            "heuristic",
        ]
    ].values:
        for block_id, budget in blocks:
            dfs.append(
                pd.DataFrame(
                    [
                        {
                            "block": block_id,
                            "budget": budget["epsilon"],
                            "mechanism": mechanism,
                            "zipf_k": zipf_k,
                            "initial_budget": initial_budget,
                            "key": key,
                            "total_tasks": total_tasks,
                            "learning_rate": learning_rate,
                            "warmup": warmup,
                            "planner": planner,
                            "heuristic": heuristic,
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
        fig.write_image(
            LOGS_PATH.joinpath(f"{experiment_path}/num_allocated_tasks.png")
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
        return fig

    def plot_budget_utilization(df, total_tasks):
        df["budget"] = df["initial_budget"] - df["budget"]
        # / df[
        # "initial_budget"
        # ]
        pivoted_df = df.pivot_table(
            columns="mechanism",
            index=[df.index.values, "zipf_k"],
            values="budget",
            aggfunc=np.sum,
        ).reset_index("zipf_k")
        pivoted_df.to_csv(
            LOGS_PATH.joinpath(f"{experiment_path}/budget_utilization.csv"),
            index=False,
        )

        fig = px.bar(
            df,
            x="zipf_k",
            y="budget",
            color="key",
            barmode="group",
            title=f"Budget Consumption - total tasks-{total_tasks}",
            width=900,
            height=500,
        )
        fig.write_image(LOGS_PATH.joinpath(f"{experiment_path}/budget_utilization.png"))
        return fig

    def plot_cumulative_budget_utilization_time(df, total_tasks):
        df.to_csv(
            LOGS_PATH.joinpath(f"{experiment_path}/cumulative_budget_utilization.csv"),
            index=False,
        )
        fig = px.line(
            df,
            x="task",
            y="cumulative_budget",
            color="key",
            title=f"Cumulative Budget Consumption - total tasks-{total_tasks}",
            # category_orders=category_orders,
            width=1000,
            height=600,
            facet_row="zipf_k",
        )
        fig.write_image(
            LOGS_PATH.joinpath(f"{experiment_path}/cumulative_budget_utilization.png")
        )
        return fig

    def plot_budget_utilization_time(df, total_tasks):
        fig = px.scatter(
            df,
            x="task",
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
    time_budgets, _ = get_budgets_information(df)
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
        grouped = df.groupby("zipf_k")
        for key, group in grouped:
            group_df = pd.DataFrame(group)
            pivoted_df = group_df.pivot_table(
                columns="mechanism",
                index=[group_df.index.values, "block"],
                values="budget",
                aggfunc=np.sum,
            ).reset_index(["block"])
            pivoted_df["block"] = pivoted_df["block"].astype(int)
            pivoted_df = pivoted_df.sort_values(["block"])
            pivoted_df.to_csv(
                LOGS_PATH.joinpath(
                    f"{experiment_path}/budget_utilization_zipf_{key}.csv"
                ),
                index=False,
            )

        fig = px.bar(
            df,
            x="block",
            y="budget",
            color="key",
            barmode="group",
            title=f"Absolute Budget Consumption - total tasks-{total_tasks}",
            width=1200,
            height=600,
            facet_row="zipf_k",
        )
        fig.write_image(LOGS_PATH.joinpath(f"{experiment_path}/budget_utilization.png"))
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

    def plot_global_budget_utilization_time(df, total_tasks):
        df["cumulative_budget"] = df["sum_budget_across_blocks"]
        df.to_csv(
            LOGS_PATH.joinpath(f"{experiment_path}/cumulative_budget_utilization.csv"),
            index=False,
        )

        fig = px.line(
            df,
            x="task",
            y="cumulative_budget",
            color="key",
            title=f"Cumulative Absolute Budget Consumption across all blocks - total tasks-{total_tasks}",
            width=1000,
            height=600,
            facet_row="zipf_k",
        )
        fig.write_image(
            LOGS_PATH.joinpath(f"{experiment_path}/cumulative_budget_utilization.png")
        )
        return fig

    df = get_df(experiment_path)
    total_tasks = df["total_tasks"].max()
    df["zipf_k"] = df["zipf_k"].astype(float)
    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)
    iplot(plot_num_allocated_tasks(df, total_tasks))
    blocks = get_blocks_information(df)
    iplot(plot_budget_utilization(blocks, total_tasks))
    iplot(plot_total_sv_checks(df, total_tasks))
    iplot(plot_total_sv_misses(df, total_tasks))

    _, global_time_budgets = get_budgets_information(df)
    iplot(plot_global_budget_utilization_time(global_time_budgets, total_tasks))


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
    # df = df.query("zipf_k == 1 and key=='Laplace+TreeCache'")

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

    iplot(plot_num_cuts(df))


# def analyze_query_types(experiment_path):
#     def plot_query_per_task(df):

#         # fig = px.histogram(
#         #     df,
#         #     x="query_id",
#         #     title="Query Pool Histograms",
#         #     # color="query_id",
#         #     width=800,
#         #     height=500,
#         #     facet_row="zipf_k",
#         # )
#         # return fig

#         fig = px.bar(
#             df,
#             x="zipf_k",
#             y="count",
#             title="Num Different queries",
#             # color="query_id",
#             width=800,
#             height=500,
#             # facet_row="zipf_k",
#         )
#         return fig

#     df = get_df(experiment_path)
#     df["zipf_k"] = df["zipf_k"].astype(float)
#     df = df.query("mechanism == 'DirectLaplace'")
#     df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)

#     tasks = get_tasks_information(df)
#     tasks = tasks[["id", "key", "query_id", "zipf_k"]]
#     print(tasks.groupby("zipf_k")["query_id"].nunique())
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
