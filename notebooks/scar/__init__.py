import csv
import os

import numpy as np


here = os.path.dirname(__file__)

ALPHA = 0.8  # class prior
C = 0.2  # label frequency

N = 10000  # number of examples

MEAN_P = [3, 3]  # the mean of the positive example's distribution
COV_P = [[1, 0], [0, 1]]  # the covariance matrix of the positive example's distribution

MEAN_N = [0, 0]  # the mean of the negative example's distribution
COV_N = [[1, 0], [0, 1]]  # the covariance matrix of the positive example's distribution


def load_scar():
    x = []
    y = []
    s = []
    with open(os.path.join(here, "scar.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append([row["x_0"], row["x_1"]])
            y.append(row["y"])
            s.append(row["s"])
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        np.asarray(s, dtype=np.int32),
    )
