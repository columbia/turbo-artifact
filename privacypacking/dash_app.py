import argparse
import dash
from collections import defaultdict

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import pprint as pp


class Plotter:
    def __init__(self, file):
        self.file = file
        with open(self.file, "r") as fp:
            log = json.load(fp)
            self.tasks = log["tasks"]
            self.blocks = log["blocks"]

    def plot(self):
        figs = []
        for block in self.blocks:
            figs.append(go.FigureWidget(self.stack_jobs_under_block_curve(block)))
        objs = []
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
                x=list(block["initial_budget"]["orders"].keys()),
                y=list(block["initial_budget"]["orders"].values()),
                name="Block capacity",
                line=dict(color="green", width=4),
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
    args = parser.parse_args()
    app = dash.Dash()

    plotter = Plotter(args.file)
    objs = plotter.plot()
    app.layout = html.Div(objs)

    app.run_server(debug=False, port="8080", host="127.0.0.1")
