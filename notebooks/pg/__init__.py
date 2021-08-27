import csv
import os

import numpy as np
from labeling import LabelingMechanism


here = os.path.dirname(__file__)

lm = None
if os.path.exists(os.path.join(here, "param.json")):
    lm = LabelingMechanism([0,1],[1,1], min_prob=0, max_prob=1)
    lm.load_param(os.path.join(here, "param.json"))

C = lm.c if lm is not None else None

ALPHA = 0.8  # class prior

N_TRAIN = 10000  # number of training examples
N_VALID = 1000  # number of validation examples
N_TEST = 1000  # number of test examples

MEAN_P = [3, 3]  # the mean of the positive example's distribution
COV_P = [[1, 0], [0, 1]]  # the covariance matrix of the positive example's distribution

MEAN_N = [0, 0]  # the mean of the negative example's distribution
COV_N = [[1, 0], [0, 1]]  # the covariance matrix of the positive example's distribution

def load_pg():
    train = _load_pg(os.path.join(here, "train.csv"))
    valid = _load_pg(os.path.join(here, "valid.csv"))
    test = _load_pg(os.path.join(here, "test.csv"))
    return train, valid, test


def _load_pg(path: str):
    x, y, s = [], [], []
    with open(path) as f:
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
