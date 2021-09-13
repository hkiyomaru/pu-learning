# Understanding PU Learning Through Implementing Its Algorithms

This repository contains notebooks that implement algorithms introduced in "[Learning from positive and unlabeled data: a survey](https://arxiv.org/abs/1811.04820)."

## Development Environment

- Python 3.9
- Dependencies: See [pyproject.toml](./pyproject.toml).

## Run Notebooks

### Online

Use [binder](https://mybinder.org/). The following link launches a Jupyter notebook server.

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

- data.ipynb: Create PU datasets that satisfy the SCAR, SAR, and PG assumptions. An example is represented as a tuple of an input vector $x$, the class $y$, the label $s$, and the propensity score $e$. Created datasets are saved in the `data` directory and used in the following notebooks.
- traditional_classifier.ipynb: Learn a traditional classifier $Pr(y=1|x)$ from a fully-labeled dataset. The performance can be seen as the upper bound that a classifier can achieve.
- non_traditional_classifier.ipynb: Learn a non-traditional classifier $Pr(s=1|x)$ from a PU dataset.

### Two-step Techniques (Section 5.1)

- two_step_spy_nb.ipynb: Learn a classifir $Pr(y=1|x)$ with a two-step technique; in the first step, reliable negative examples are found by Spy; in the second step, a naive bayes classifier is trained.
- two_step_1dnf_itersvm.ipynb: Learn a classifir $Pr(y=1|x)$ with a two-step technique; in the first step, reliable negative examples are found by 1-DNF; in the second step, an iterative SVM is trained.

### Biased Learning (Section 5.2)

- biased_svm.ipynb: Learn a classifir $Pr(y=1|x)$ by biased SVM that penalizes misclassified positive and negative examples
differently; the weight is determined according to $F1'$.

### Incorporation of the Class Prior (Section 5.3)

- postprocessing.ipynb: Calculate $Pr(y=1|x)$ by scaling the prediciton of a non-traditional classifier $Pr(s=1|x)$ according to the label frequency $c$.
- duplication.ipynb: Create a new dataset from PU data so that a classifier learned on it is expected to be equal to a classifier trained from a fully labeled dataset, and then learn a classifier $Pr(s=1|x)$ on it.
- empirical_risk_minimization.ipynb: Create a new dataset from PU data so that a classifier learned on it is expected to be equal to a classifier trained from a fully labeled dataset, and then learn a classifier $Pr(s=1|x)$ on it.

## Reference

- [Learning from positive and unlabeled data: a survey (Jessa Bekker and Jesse Davis, 2020)](https://arxiv.org/abs/1811.04820)
