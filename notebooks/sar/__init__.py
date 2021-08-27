import csv
import os

import numpy as np


here = os.path.dirname(__file__)

ALPHA = 0.8  # class prior

N_TRAIN = 10000  # number of training examples
N_VALID = 1000  # number of validation examples
N_TEST = 1000  # number of test examples

MEAN_P = [3, 3]  # the mean of the positive example's distribution
COV_P = [[1, 0], [0, 1]]  # the covariance matrix of the positive example's distribution

MEAN_N = [0, 0]  # the mean of the negative example's distribution
COV_N = [[1, 0], [0, 1]]  # the covariance matrix of the positive example's distribution