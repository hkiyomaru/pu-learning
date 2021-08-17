# Understanding PU Learning through Implementating its Algorithms

## Development Environment

- Python 3.9
- Dependencies: See [pyproject.toml](./pyproject.toml).

## Installation

Use [poetry](https://python-poetry.org/).

```
$ poetry install
```

## Notebooks

### Creating a dataset

- [scar](./notebooks/scar.ipynb): Creates a dataset that satisfies the SCAR assumption. The parameters are written in [./notebooks/scar/__init__.py](./notebooks/scar/__init__.py).

### Training a classifier

- [traditional_classifier](./noteboooks/traditional_classifier.ipynb): Learns a traditional classifier in the standard binary classification setting.
- [nontraditional_classifier](./noteboooks/nontraditional_classifier.ipynb): Learns a non-traditional classifier.
- [postprocessing](./notebooks/postprocessing.ipynb): Learns a traditional classifier from PU data by postprocessing the output of a non-traditional classifier.
- [rebalancing](./notebooks/rebalancing.ipynb): Learns a traditional classifier from PU data by a rebalancing method.
- [labeling_probability](notebooks/.ipynb_checkpoints/labeling_probability.ipynb): Learns a traditional classifier from PU data by incorporating labeling probabilities estimated by a non-traditional classifier.
- [empirical_risk_minimization](./notebooks/empirical_risk_minimization.ipynb): Learns a traditional classifier from PU data based on empirical risk minimization.
