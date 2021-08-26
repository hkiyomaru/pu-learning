import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score

POSITIVE_COLOR = "#1E90FF"
NEGATIVE_COLOR = "#FF6347"
UNLABELED_COLOR = "#A9A9A9"


def plot_x_y(x: np.array, y: np.array):
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
        color_discrete_map={"Negative": NEGATIVE_COLOR, "Positive": POSITIVE_COLOR},
    )


def plot_x_y_proba(x: np.array, y_proba: np.array):
    df = pd.DataFrame.from_dict(
        {
            "x_0": x[:, 0],
            "x_1": x[:, 1],
            "y": y_proba,
        }
    )
    return px.scatter(
        df,
        x="x_0",
        y="x_1",
        color="y",
        color_continuous_scale=[NEGATIVE_COLOR, POSITIVE_COLOR],
        range_color=[0.0, 1.0],
    )


def plot_x_s(x: np.array, s: np.array):
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
        color_discrete_map={"Unlabeled": UNLABELED_COLOR, "Positive": POSITIVE_COLOR},
    )


def plot_x_s_proba(x: np.array, s_proba: np.array):
    df = pd.DataFrame.from_dict(
        {
            "x_0": x[:, 0],
            "x_1": x[:, 1],
            "s": s_proba,
        }
    )
    return px.scatter(
        df,
        x="x_0",
        y="x_1",
        color="s",
        color_continuous_scale=[UNLABELED_COLOR, POSITIVE_COLOR],
        range_color=[0.0, 1.0],
    )


def f1_prime(y: np.array, y_hat: np.array):
    r = recall_score(y, y_hat)
    ratio_p = len(y_hat[y_hat == 1]) / len(y_hat)
    if ratio_p == 0.0:
        return 0.0
    else:
        return r ** 2 / ratio_p
