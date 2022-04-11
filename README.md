# Understanding PU Learning Through Implementing Its Algorithms

This repository contains notebooks that implement algorithms introduced in "[Learning from positive and unlabeled data: a survey](https://arxiv.org/abs/1811.04820)."

## Development Environment

- Python 3.9
- Dependencies: See [requirements.txt](./requirements.txt).

## Getting Started

### Online

Use [binder](https://mybinder.org/) to start a Jupyter notebook server online.

- [https://mybinder.org/v2/gh/hkiyomaru/pu-learning/HEAD](https://mybinder.org/v2/gh/hkiyomaru/pu-learning/HEAD)

### Local

Install the dependencies.

```
$ pip install -r requirements.txt
```

Then launch a Jupyter notebook server.

```
$ jupyter lab
```

## Notebooks

### PU datasets (Section 3.1)

- **data.ipynb**: This notebook creates PU datasets that satisfy the SCAR, SAR, and PG assumptions. Created datasets are saved in the `data` directory and used in the following notebooks.
- **traditional_classifier.ipynb**: This notebook learns a traditional classifier from a fully-labeled dataset. The performance can be seen as the upper bound that a classifier can achieve.
- **non_traditional_classifier.ipynb**: This notebook learns a non-traditional classifier from a PU dataset.

### Two-step Techniques (Section 5.1)

- **two_step_spy_nb.ipynb**: This notebook learns a classifir with a two-step technique; in the first step, reliable negative examples are found by *spy*; in the second step, a naive bayes classifier is trained.
- **two_step_1dnf_itersvm.ipynb**: This notebook learns a classifir with a two-step technique; in the first step, reliable negative examples are found by 1-DNF; in the second step, an iterative SVM is trained.

### Biased Learning (Section 5.2)

- **biased_svm.ipynb**: This notebook learns a biased SVM that penalizes misclassified positive (labeled) and negative (unlabeled) examples differently; the weight is determined according to F1'.

### Incorporation of the Class Prior (Section 5.3)

- **postprocessing.ipynb**: This notebook predicts the probability of an example being positive by scaling the prediciton of a non-traditional classifier according to the label frequency.
- **duplication.ipynb**: This notebook first creates a new dataset from PU data so that a classifier learned on it is expected to be equal to a classifier trained from a fully labeled dataset, and then learns a classifier on it. This method assumes that the PU data meets the SCAR assumption.
- **empirical_risk_minimization.ipynb**: This notebook first creates a new dataset from PU data so that a classifier learned on it is expected to be equal to a classifier trained from a fully labeled dataset, and then learn a classifier on it. This method does not put any assumption on the PU data.

## Authors

- Hirokazu Kiyomaru (@hkiyomaru)
- Yukiya Wada (@YukiyaWada)
- Nozomu Karai (@nozomu-karai)

## Reference

- [Learning from positive and unlabeled data: a survey (Jessa Bekker and Jesse Davis, 2020)](https://arxiv.org/abs/1811.04820)
