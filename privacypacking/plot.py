from collections import defaultdict

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


def stack_jobs_under_block_curve(job_list, block, allocation_status_list):
    data = defaultdict(list)
    for i, (job, status) in enumerate(zip(job_list, allocation_status_list)):
        for alpha, epsilon in job.orders.items():
            data["alpha"].append(alpha)
            data["epsilon"].append(epsilon)
            data["job"].append(i)
            data["allocated"].append(status)

    df = pd.DataFrame(data=data)
    df = df.sort_values(by=["allocated"], ascending=False)

    fig = px.area(
        df,
        x="alpha",
        y="epsilon",
        color="allocated",
        line_group="job",
        log_x=True,
        log_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=list(block.orders.keys()),
            y=list(block.orders.values()),
            name="Block capacity",
            line=dict(color="green", width=4),
        )
    )

    log_toggle(fig)

    fig.show()
    # return fig
