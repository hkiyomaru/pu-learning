# Understanding PU Learning Through Implementing Its Algorithms

This repository contains notebooks that implement algorithms introduced in "[Learning from positive and unlabeled data: a survey](https://arxiv.org/abs/1811.04820)."

## Development Environment

- Python 3.9
- Dependencies: See [pyproject.toml](./pyproject.toml).

## Run Notebooks

### Online

Use [binder](https://mybinder.org/) as an online Jupyter notebook environment.

- [https://mybinder.org/v2/gh/hkiyomaru/pu-learning/HEAD](https://mybinder.org/v2/gh/hkiyomaru/pu-learning/HEAD)

### Local

Use [poetry](https://python-poetry.org/) to install the dependencies.

```
$ poetry install
```

Then launch a Jupyter notebook server.

```
$ poetry run jupyter lab
```

## Notebooks

### PU datasets (Section 3.1)

- data.ipynb: Create PU datasets that satisfy the SCAR, SAR, and PG assumptions. An example is a tuple of an input vector, the class, the label, and the propensity score. Created datasets are saved in the `data` directory and used in the following notebooks.
- traditional_classifier.ipynb: Learn a traditional classifier from a fully-labeled dataset. The performance can be seen as the upper bound that a classifier can achieve.
- non_traditional_classifier.ipynb: Learn a non-traditional classifier from a PU dataset.

### Two-step Techniques (Section 5.1)

- two_step_spy_nb.ipynb: Learn a classifir with a two-step technique; in the first step, reliable negative examples are found by Spy; in the second step, a naive bayes classifier is trained.
- two_step_1dnf_itersvm.ipynb: Learn a classifir with a two-step technique; in the first step, reliable negative examples are found by 1-DNF; in the second step, an iterative SVM is trained.

### Biased Learning (Section 5.2)

- biased_svm.ipynb: Learn a classifir by biased SVM that penalizes misclassified positive and negative examples
differently; the weight is determined according to F1'.

### Incorporation of the Class Prior (Section 5.3)

- postprocessing.ipynb: Calculate by scaling the prediciton of a non-traditional classifier according to the label frequency.
- duplication.ipynb: Create a new dataset from PU data so that a classifier learned on it is expected to be equal to a classifier trained from a fully labeled dataset, and then learn a classifier on it.
- empirical_risk_minimization.ipynb: Create a new dataset from PU data so that a classifier learned on it is expected to be equal to a classifier trained from a fully labeled dataset, and then learn a classifier on it.

## Reference

- [Learning from positive and unlabeled data: a survey (Jessa Bekker and Jesse Davis, 2020)](https://arxiv.org/abs/1811.04820)
