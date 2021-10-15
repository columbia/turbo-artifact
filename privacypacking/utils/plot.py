from collections import defaultdict
from typing import List, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.missing_ipywidgets import FigureWidget

from privacypacking.budget import Budget


def plot_budgets(budgets: Union[List[Budget], Budget]) -> FigureWidget:
    if isinstance(budgets, Budget):
        budgets = [budgets]

    data = defaultdict(list)
    for i, budget in enumerate(budgets):
        for alpha, epsilon in zip(budget.alphas, budget.epsilons):
            data["alpha"].append(alpha)
            data["epsilon"].append(epsilon)
            data["id"].append(i)

    df = pd.DataFrame(data=data)
    if not df.empty:
        fig = px.line(
            df,
            x="alpha",
            y="epsilon",
            color="id",
            log_x=False,
            log_y=True,
        )
    else:
        fig = px.area(
            log_x=False,
            log_y=True,
        )

    return fig
