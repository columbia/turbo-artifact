import argparse
import json
from datetime import datetime
from collections import defaultdict

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Output, Input
from loguru import logger

from privacypacking.utils.utils import LOGS_PATH


class Plotter:
    def __init__(self, file):
        self.file = file
        with open(self.file, "r") as fp:
            log = json.load(fp)
            self.scheduler_name = log["scheduler_name"]
            self.num_scheduled_tasks = log["num_scheduled_tasks"]
            self.tasks = log["tasks"]
            self.blocks = log["blocks"]

    def plot(self):
        figs = []
        for block in self.blocks:
            figs.append(go.FigureWidget(self.stack_jobs_under_block_curve(block)))
        objs = [
            html.Div(
                html.H1(f"Scheduler: {self.scheduler_name}"), className="six columns"
            )
        ]
        for i, fig in enumerate(figs):
            objs += [
                html.Div(
                    [html.H3(f"Block {i + 1}"), dcc.Graph(id=f"g{i}", figure=fig)],
                    className="six columns",
                )
            ]

        return objs

    def stack_jobs_under_block_curve(self, block):
        data = defaultdict(list)
        for task in self.tasks:
            block_id = str(block["id"])
            if block_id in task["budget_per_block"]:
                for alpha, epsilon in task["budget_per_block"][block_id][
                    "orders"
                ].items():
                    data["alpha"].append(alpha)
                    data["epsilon"].append(epsilon)
                    data["job"].append(task["id"])
                    data["allocated"].append(task["allocated"])
                    data["dp_epsilon"].append(
                        task["budget_per_block"][block_id]["dp_budget"]["epsilon"]
                    )

        df = pd.DataFrame(data=data)
        if not df.empty:
            df = df.sort_values(by=["allocated", "dp_epsilon"], ascending=[False, True])
            fig = px.area(
                df,
                x="alpha",
                y="epsilon",
                color="allocated",
                line_group="job",
                log_x=False,
                log_y=True,
            )
        else:
            fig = px.area(
                log_x=False,
                log_y=True,
            )

        fig.add_trace(
            go.Scatter(
                x=list(block["initial_budget"]["orders"].keys()),
                y=list(block["initial_budget"]["orders"].values()),
                name="Block capacity",
                line=dict(color="green", width=4),
            )
        )

        # Temporary hack to plot the number of scheduled tasks (don't know how plotly works)
        fig.add_trace(
            go.Scatter(
                x=[1.5],
                y=[0],
                name=f"Scheduled Jobs: {self.num_scheduled_tasks}",
                line=dict(color="black", width=1),
            )
        )

        self.log_toggle(fig)
        return fig

    def log_toggle(self, fig):
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label="Linear",
                            method="relayout",
                            args=[{"yaxis.type": "linear"}],
                        ),
                        dict(
                            label="Log", method="relayout", args=[{"yaxis.type": "log"}]
                        ),
                    ]
                )
            ],
        )

    def save_fig(self, plotly_fig, filename="plot.png"):
        plotly_fig.write_image(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", dest="file")
    parser.add_argument("--port", dest="port", default="8080")
    args = parser.parse_args()

    if args.file:
        file = args.file
    else:
        logs = LOGS_PATH.glob("**/*.json")
        most_recent_date = datetime.min
        most_recent_log = None
        for log in logs:
            date = datetime.strptime(log.stem, "%m%d-%H%M%S")
            if date > most_recent_date:
                most_recent_log = log
                most_recent_date = date
        file = most_recent_log

        logger.info(f"No file provided. Plotting the most recent experiment: {file}")

    app = dash.Dash()
    app.layout = html.Div(
        html.Div(
            [
                html.Div(id="live-update-text"),
                dcc.Interval(
                    id="interval-component",
                    interval=5 * 1000,  # in milliseconds
                    n_intervals=0,
                ),
            ]
        )
    )

    @app.callback(
        Output("live-update-text", "children"),
        Input("interval-component", "n_intervals"),
    )
    def update(n):
        objs = Plotter(file).plot()
        return html.Div(objs)

    app.run_server(debug=False, port=args.port, host="127.0.0.1")
