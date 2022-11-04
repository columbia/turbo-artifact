from collections import defaultdict
from typing import Dict, List, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.missing_ipywidgets import FigureWidget

from privacypacking.budget import Budget, RenyiBudget


def df_normalized_curves(curves: Dict[str, Budget], epsilon=10, delta=1e-6):
    d = defaultdict(list)
    block = RenyiBudget.from_epsilon_delta(epsilon=epsilon, delta=delta)
    for name, curve in curves.items():
        normalized_curve = curve.normalize_by(block)
        normalized_curve, curve = RenyiBudget.same_support(normalized_curve, curve)
        d["alpha"].extend(curve.alphas)
        d["rdp_epsilon"].extend(curve.epsilons)
        d["normalized_rdp_epsilon"].extend(normalized_curve.epsilons)
        d["mech_type"].extend([name.split("-")[0]] * len(curve.alphas))
        d["mech_name"].extend([name] * len(curve.alphas))
    return pd.DataFrame(d)


def plot_budgets(
    budgets: Union[List[Budget], Budget], log_x=False, log_y=False
) -> FigureWidget:
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
            log_x=log_x,
            log_y=log_y,
        )
    else:
        fig = px.area(
            log_x=log_x,
            log_y=log_y,
        )

    return fig
