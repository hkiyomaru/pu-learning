import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import recall_score

POSITIVE_COLOR = "#1E90FF"
NEGATIVE_COLOR = "#FF6347"
LABELED_COLOR = "#78B639"
UNLABELED_COLOR = "#A9A9A9"


def plot_x_y(xs: np.array, ys: np.array):
    df = pd.DataFrame.from_dict(
        {
            "x_0": xs[:, 0],
            "x_1": xs[:, 1],
            "y": ["Positive" if y == 1 else "Negative" for y in ys],
        }
    )
    return px.scatter(
        df,
        x="x_0",
        y="x_1",
        color="y",
        color_discrete_map={"Negative": NEGATIVE_COLOR, "Positive": POSITIVE_COLOR},
    )


def plot_x_y_proba(xs: np.array, ys_prob: np.array):
    df = pd.DataFrame.from_dict(
        {
            "x_0": xs[:, 0],
            "x_1": xs[:, 1],
            "y": ys_prob,
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


def plot_x_y_list(xs: np.array, ys_list: list):
    df = pd.DataFrame(index=[], columns=["x_0", "x_1", "y", "clf_index"])
    for i in range(len(ys_list)):
        df_ = pd.DataFrame.from_dict(
            {
                "x_0": xs[:, 0],
                "x_1": xs[:, 1],
                "y": ["Positive" if y == 1 else "Negative" for y in ys_list[i]],
                "clf_index": i,
            }
        )
        df = pd.concat([df, df_])
    return px.scatter(
        df,
        x="x_0",
        y="x_1",
        facet_row="clf_index",
        color="y",
        color_discrete_map={"Negative": NEGATIVE_COLOR, "Positive": POSITIVE_COLOR},
    )


def plot_x_s(xs: np.array, ss: np.array):
    df = pd.DataFrame.from_dict(
        {
            "x_0": xs[:, 0],
            "x_1": xs[:, 1],
            "s": ["Positive" if s == 1 else "Unlabeled" for s in ss],
        }
    )
    return px.scatter(
        df,
        x="x_0",
        y="x_1",
        color="s",
        color_discrete_map={"Unlabeled": UNLABELED_COLOR, "Positive": LABELED_COLOR},
    )


def plot_x_s_proba(xs: np.array, ss_prob: np.array):
    df = pd.DataFrame.from_dict(
        {
            "x_0": xs[:, 0],
            "x_1": xs[:, 1],
            "s": ss_prob,
        }
    )
    return px.scatter(
        df,
        x="x_0",
        y="x_1",
        color="s",
        color_continuous_scale=[UNLABELED_COLOR, LABELED_COLOR],
        range_color=[0.0, 1.0],
    )


def f1_prime(ys: np.array, ys_hat: np.array):
    r = recall_score(ys, ys_hat)
    ratio_p = len(ys_hat[ys_hat == 1]) / len(ys_hat)
    if ratio_p == 0.0:
        return 0.0
    else:
        return r**2 / ratio_p
