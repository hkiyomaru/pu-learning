import csv
import os

import numpy as np

here = os.path.dirname(__file__)


def load_scar():
    return _load(os.path.join(here, "scar"))


def load_sar():
    return _load(os.path.join(here, "sar"))


def load_pg():
    return _load(os.path.join(here, "pg"))
    

def _load(path: str):
    train = _load_data(os.path.join(path, "train.csv"))
    valid = _load_data(os.path.join(path, "valid.csv"))
    test = _load_data(os.path.join(path, "test.csv"))
    c = _load_label_frequency(os.path.join(path, "c.txt"))
    return train, valid, test, c



def _load_data(path: str):
    xs, ys, ss, es = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append([row["x_0"], row["x_1"]])
            ys.append(row["y"])
            ss.append(row["s"])
            es.append(row["e"])
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int32),
        np.asarray(ss, dtype=np.int32),
        np.asarray(es, dtype=np.float32),
    )


def _load_label_frequency(path: str):
    with open(path) as f:
        return float(f.read())
