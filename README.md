# Ensemble Methods

Code for combining decision trees into stronger models: bootstrap sampling, majority voting, bagging, random forests, boosting, AdaBoost and gradient boosting.

---

## Theory

Lecture notes for this topic are coming soon on https://parag1604.github.io/notes

The trees used as building blocks are written from scratch in the companion repository [Decision-Trees](https://github.com/parag1604/Decision-Trees); here we use scikit-learn's trees and focus on how they are trained and combined.

## Python Environment Setup

Use Python >= 3.10 and install the requirements:

```bash
pip install -r requirements.txt
```

- Numpy, Scipy - vectorized operations and the binomial distribution
- Pandas - for reading and encoding the CSV files
- Scikit-learn - decision trees used as base learners
- Matplotlib - required for plotting

## Data

Scripts 01 and 02 need no data. The others expect the same two CSV files as the Decision-Trees repository, under `./data` (pass other paths with `--car_path`, `--house_path` or `--data_path`):

```
data/
  car.csv     # classification (03, 04)
  house.csv   # regression (03, 04 with --task reg, and 05-07)
```

**car.csv**: the [UCI Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation) dataset. Download `car.data` and save it as `data/car.csv` (a header row is optional; columns in the UCI order: buying price, maintenance cost, doors, number of persons, lug_boot, safety, decision). The target is acceptable vs. unacceptable.

**house.csv**: the Kaggle [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) (`Housing.csv`), saved as `data/house.csv`, with the columns `price, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus`. Prices are standardized with the training statistics, so an RMSE of 1.0 is no better than always predicting the mean price.

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
