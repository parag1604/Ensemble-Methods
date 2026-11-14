# Ensemble Methods

Code for combining decision trees into stronger models: bootstrap sampling, majority voting, bagging, random forests, boosting, AdaBoost and gradient boosting.

Everything is implemented **from scratch** in numpy, including the base learners: `tree.py` is a CART implementation (classification and regression trees with `max_depth`, `min_samples_split`, `max_features` and `sample_weight`) and `model.py` builds the ensembles on top of it. scikit-learn is used only in `08_sklearn_comparison.py`, to check our results against the library versions.

---

## Theory

Lecture notes for this topic are coming soon on https://parag1604.github.io/notes

The trees in `tree.py` are adapted from the companion repository [Decision-Trees](https://github.com/parag1604/Decision-Trees), which explains how a tree is grown one threshold at a time. This version finds the same splits with a vectorised search (sort each feature once, then score every threshold with cumulative sums) and adds the options the ensembles need: `max_features` for random forests and `sample_weight` for boosting.

## Python Environment Setup

Use Python >= 3.10 and install the requirements:

```bash
pip install -r requirements.txt
```

- Numpy, Scipy - vectorized operations and the binomial distribution
- Pandas - for reading and encoding the CSV files
- Scikit-learn - only for the reference models in `08_sklearn_comparison.py`
- Matplotlib - required for plotting

## Data

Scripts 03–08 use the same two small CSV files as the Decision-Trees repository, which are **included** under `./data` (pass other paths with `--car_path`, `--house_path` or `--data_path`):

```
data/
  car.csv     # classification (03, 04)
  house.csv   # regression (03, 04 with --task reg, and 05-07)
```

**car.csv**: the [UCI Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation) dataset. It is included as `data/car.csv` (`5more` doors is written `5`; with the original `car.data` a header row is optional; columns in the UCI order: buying price, maintenance cost, doors, number of persons, lug_boot, safety, decision). The target is acceptable vs. unacceptable.

**No data?** Add `--synthetic` to scripts 03–08 to use numpy-generated stand-ins instead: a car-like classification set (1728 points, 6 ordinal features, a non-linear rule with 10% label noise) and the Friedman #1 regression problem (545 points, 12 features of which 5 matter, Gaussian noise).

**house.csv**: the Kaggle [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) (`Housing.csv`), included as `data/house.csv`, with the columns `price, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus`. Prices are standardized with the training statistics, so an RMSE of 1.0 is no better than always predicting the mean price.

## Running

### 1. Unique points in a bootstrap sample

```bash
python 01_bootstrap_unique_points.py --n 1000000
```

A bootstrap sample (N draws with replacement) contains only about 63.2% (1 - 1/e) of the distinct points:

```bash
Unique data points: 632047 (63.20%)
Exact expectation 1 - (1 - 1/n)^n: 63.21%
Limit 1 - 1/e:                     63.21%
```

### 2. Majority vote of weak learners

```bash
python 02_majority_vote.py
```

Plots the accuracy of a majority vote of n independent learners, each only slightly better than chance. With 201 learners that are 51%, 55% and 60% accurate, the vote is 61%, 92% and 99.8% accurate. The assumption of independent errors is what bagging and random forests try to approximate.

### 3. Bagging

```bash
python 03_bagging.py --task clf
python 03_bagging.py --task reg
```

Trains 21 trees of depth 8 on bootstrap samples and combines them (majority vote / average). Compare the test score with that of a single tree of the same depth.

### 4. Random forest

```bash
python 04_random_forest.py --task clf
python 04_random_forest.py --task reg --max_features 0.33
```

New: each split only considers a random subset of the features (`sqrt` of the number of features for classification, a third for regression by default), which decorrelates the trees. `--max_features 1.0` gives back plain bagging.

### 5. Boosting, the intuition

```bash
python 05_boosting.py --max_depth 3 --n_estimators 20
```

New: trees are trained one after the other, each with sample weights proportional to the ensemble's current squared error, and averaged. This is a heuristic that shows the idea of "focus on the mistakes"; the next two scripts are the standard algorithms.

### 6. AdaBoost (AdaBoost.R2)

```bash
python 06_adaptive_boosting.py --max_depth 3 --n_estimators 20 --loss linear
```

New: multiplicative weight updates, a per-tree importance `log(1/beta)` based on its weighted loss, early stopping when a tree's average loss reaches 0.5, and a weighted-median prediction.

### 7. Gradient boosting

```bash
python 07_gradient_boosting.py --max_depth 3 --n_estimators 20 --learning_rate 0.2
```

New: each tree is fit to the residuals of the current ensemble (the negative gradient of the squared loss) and added with a small step. The script also prints the train and test RMSE as trees are added, so you can see where more trees stop helping.

Each boosting script prints the RMSE of a single tree of the same depth next to the ensemble's, for comparison.

### 8. Comparison with scikit-learn

```bash
python 08_sklearn_comparison.py            # car.csv and house.csv
python 08_sklearn_comparison.py --synthetic
```

New: the only script that uses library models. It trains our trees, bagging, random forests, AdaBoost.R2 and gradient boosting next to scikit-learn's `DecisionTree*`, `RandomForest*`, `AdaBoostRegressor` and `GradientBoostingRegressor` with the same settings, and prints the test scores and training times.

## What was run

All scripts were run on a MacBook CPU on the included car.csv and house.csv; each takes under 4 s. Test scores (accuracy on car.csv, standardized RMSE on house.csv):

| script | single tree (test) | ensemble (test) |
|---|---|---|
| 03 bagging, classification (accuracy) | 0.980 | 0.994 |
| 03 bagging, regression (RMSE) | 0.844 | 0.641 |
| 04 random forest, classification (accuracy) | 0.980 | 0.980 |
| 04 random forest, regression (RMSE) | 0.844 | 0.617 |
| 05 boosting heuristic (RMSE, depth 3) | 0.762 | 0.689 |
| 06 AdaBoost.R2 (RMSE, depth 3) | 0.762 | 0.691 |
| 07 gradient boosting (RMSE, depth 3) | 0.762 | 0.607 |

`08_sklearn_comparison.py` puts the from-scratch models next to scikit-learn's:

| model | ours | scikit-learn |
|---|---|---|
| single tree (accuracy) | 0.980 | 0.980 |
| bagging, 21 trees (accuracy) | 0.994 | 0.988 |
| random forest, 21 trees (accuracy) | 0.980 | 0.983 |
| single tree (RMSE) | 0.844 | 0.806 |
| bagging, 21 trees (RMSE) | 0.641 | 0.646 |
| random forest, 21 trees (RMSE) | 0.617 | 0.660 |
| AdaBoost.R2, 20 trees (RMSE) | 0.691 | 0.698 |
| gradient boosting, 20 trees (RMSE) | 0.607 | 0.607 |

Gradient boosting matches scikit-learn exactly. The other differences come from randomness (bootstrap samples and feature subsets) and from tie-breaking between equally good splits deep in a tree, where we keep the first such feature and scikit-learn picks one at random. Our numpy trees are 10–20× slower than scikit-learn's compiled ones, which is still under 4 s here. The `--synthetic` stand-in data (see Data) gives the same qualitative picture.
