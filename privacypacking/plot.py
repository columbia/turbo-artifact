import pickle
from collections import defaultdict

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class Plotter:
    def __init__(self, file):
        self.file = file

    def plot(self, tasks, blocks, allocation):
        figs = []
        for k, block in enumerate(blocks):
            figs.append(
                go.FigureWidget(
                    self.stack_jobs_under_block_curve(
                        [task.budget_per_block[k] for task in tasks], block, allocation
                    )
                )
            )
        objs = []
        for i, fig in enumerate(figs):
            objs += [
                html.Div(
                    [html.H3(f"Block {i + 1}"), dcc.Graph(id=f"g{i}", figure=fig)],
                    className="six columns",
                )
            ]

        # Pickle the figures in a file so that the dash server can show it in browser
        with open(self.file, "wb") as fp:
            pickle.dump(objs, fp)

    def stack_jobs_under_block_curve(self, job_list, block, allocation_status_list):
        data = defaultdict(list)
        for i, (job, status) in enumerate(zip(job_list, allocation_status_list)):
            for alpha, epsilon in zip(job.alphas, job.epsilons):
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
                x=list(block.initial_budget.alphas),
                y=list(block.initial_budget.epsilons),
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
