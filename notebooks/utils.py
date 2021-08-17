import plotly.express as px
import pandas as pd
import numpy as np


def visualize_pn_data(x: np.array, y: np.array):
    df = pd.DataFrame.from_dict(
        {
            "x_0": x[:, 0],
            "x_1": x[:, 1],
            "y": ["Positive" if y_i == 1 else "Negative" for y_i in y],
        }
    )
    return px.scatter(
        df,
        x="x_0",
        y="x_1",
        color="y",
        color_discrete_map={"Positive": "#1E90FF", "Negative": "#FF6347"},
    )


def visualize_pu_data(x: np.array, s: np.array):
    df = pd.DataFrame.from_dict(
        {
            "x_0": x[:, 0],
            "x_1": x[:, 1],
            "s": ["Positive" if s_i == 1 else "Unlabeled" for s_i in s],
        }
    )
    return px.scatter(
        df,
        x="x_0",
        y="x_1",
        color="s",
        color_discrete_map={"Positive": "#1E90FF", "Unlabeled": "#A9A9A9"},
    )
