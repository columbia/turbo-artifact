from collections import defaultdict

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def save_fig(plotly_fig, filename="plot.png"):
    plotly_fig.write_image(filename)


def log_toggle(fig):
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="Linear",
                        method="relayout",
                        args=[{"yaxis.type": "linear"}],
                    ),
                    dict(label="Log", method="relayout", args=[{"yaxis.type": "log"}]),
                ]
            )
        ],
    )


def plot(tasks, blocks, allocation):
    figs = []
    for k, block in enumerate(blocks):
        figs.append(
            go.FigureWidget(
                stack_jobs_under_block_curve(
                    [task.budget_per_block[k] for task in tasks], block, allocation
                )
            )
        )
    app = dash.Dash()
    objs = []
    for i, fig in enumerate(figs):
        objs += [
            html.Div(
                [html.H3(f"Block {i + 1}"), dcc.Graph(id=f"g{i}", figure=fig)],
                className="six columns",
            )
        ]

    app.layout = html.Div(objs)

    app.run_server(debug=False, port="8080", host="127.0.0.1")


def stack_jobs_under_block_curve(job_list, block, allocation_status_list):
    data = defaultdict(list)
    for i, (job, status) in enumerate(zip(job_list, allocation_status_list)):
        for alpha, epsilon in job.orders.items():
            data["alpha"].append(alpha)
            data["epsilon"].append(epsilon)
            data["job"].append(i)
            data["allocated"].append(status)
            data["dp_epsilon"].append(job.dp_budget().epsilon)

    df = pd.DataFrame(data=data)
    df = df.sort_values(by=["allocated", "dp_epsilon"], ascending=[False, True])

    fig = px.area(
        df,
        x="alpha",
        y="epsilon",
        color="allocated",
        line_group="job",
        log_x=False,
        log_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=list(block.budget.orders.keys()),
            y=list(block.budget.orders.values()),
            name="Block capacity",
            line=dict(color="green", width=4),
        )
    )

    log_toggle(fig)

    return fig
