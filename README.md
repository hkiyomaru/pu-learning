# PU Learning Algorithms: Implementation Collection

This repository contains a collection of notebooks that implement algorithms discussed in the paper "[Learning from positive and unlabeled data: a survey](https://arxiv.org/abs/1811.04820)."

**Disclaimer**: This is not the official implementation. Although we carefully implemented the algorithms, we cannot guarantee that the implementation is correct. If you find any bugs, please let us know by creating an issue.

## Requirements

- Python 3.9
- Dependencies: See [requirements.txt](./requirements.txt).

## Getting Started

### Online

You can quickly launch the Jupyter notebook server online using [Binder](https://mybinder.org/).

- [https://mybinder.org/v2/gh/hkiyomaru/pu-learning/HEAD](https://mybinder.org/v2/gh/hkiyomaru/pu-learning/HEAD)

### Local

1. Install the required dependencies by executing the following command:

```
$ pip install -r requirements.txt
```

2. Launch the Jupyter notebook server:

```
$ jupyter lab
```

## Notebooks

### PU dataset (Section 3.1)

- **data.ipynb**: This notebook generates PU datasets that satisfy the SCAR, SAR, and PG assumptions. The created datasets are saved in the `data` directory for further usage in subsequent notebooks.
- **traditional_classifier.ipynb**: This notebook trains a traditional classifier using a fully-labeled dataset. It provides a performance benchmark representing the upper bound achievable by a classifier.
- **non_traditional_classifier.ipynb**: This notebook trains a non-traditional classifier using a PU dataset.

### Two-step Technique (Section 5.1)

- **two_step_spy_nb.ipynb**: This notebook trains a classifir with a two-step technique. In the first step, reliable negative examples are identified by *spy*. In the second step, a naive bayes classifier is trained.
- **two_step_1dnf_itersvm.ipynb**: This notebook demonstrates a two-step technique where reliable negative examples are selected using 1-DNF, followed by training an iterative SVM.

### Biased Learning (Section 5.2)

- **biased_svm.ipynb**: This notebook trains a biased SVM that penalizes misclassified positive (labeled) and negative (unlabeled) examples differently. The weight is determined according to F1'.

### Incorporation of the Class Prior (Section 5.3)

- **postprocessing.ipynb**: This notebook predicts the probability of an example being positive by scaling the prediciton of a non-traditional classifier according to the label frequency.
- **duplication.ipynb**: This notebook creates a new dataset from a PU dataset, allowing a classifier trained on it to be equivalent to the one trained from a fully labeled dataset. This method assumes that the PU data satisfies the SCAR assumption.
- **empirical_risk_minimization.ipynb**: This notebook creates a new dataset from a PU dataset to enable training a classifier that is expected to be equivalent to the one trained from a fully labeled dataset. This method does not impose any assumptions on the PU data.

## Authors

- Hirokazu Kiyomaru ([@hkiyomaru](https://github.com/hkiyomaru))
- Yukiya Wada ([@YukiyaWada](https://github.com/YukiyaWada))
- Nozomu Karai ([@nozomu-karai](https://github.com/nozomu-karai))

## Reference

- [Learning from positive and unlabeled data: a survey (Jessa Bekker and Jesse Davis, 2020)](https://arxiv.org/abs/1811.04820)
