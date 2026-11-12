"""08: our from-scratch ensembles next to scikit-learn's, as a sanity check.

This is the ONLY file in the repository that uses scikit-learn models.
Everything in tree.py and model.py is written from scratch; here we train the
library versions with the same settings and print both scores side by side.

- Single tree, bagging, random forest  (classification on car.csv)
- Single tree, bagging, random forest  (regression on house.csv)
- AdaBoost.R2 and gradient boosting    (regression on house.csv)

The scores should be close but not identical: the ensembles are random
(bootstrap samples, feature subsets, AdaBoost resampling), and scikit-learn
draws its random numbers differently. For gradient boosting, which has no
randomness here, the two should agree almost exactly.
(The boosting heuristic of 05 is not a standard algorithm, so it has no
library counterpart.)

Run: python 08_sklearn_comparison.py            (needs car.csv and house.csv)
     python 08_sklearn_comparison.py --synthetic
"""
import argparse
import time

import numpy as np
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn import tree as sk_tree

import tree
from model import RandomForest, AdaptiveBoostingEnsemble, GradientBoostingEnsemble
from utils import get_data, rmse


def score(model, X_train, X_test, y_train, y_test, regression):
    t = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t
    pred = model.predict(X_test)
    s = rmse(pred, y_test) if regression else float(np.mean(pred == y_test))
    return s, elapsed


def row(name, ours, theirs, args, regression):
    data = (get_data(args, "reg" if regression else "clf"))
    s1, t1 = score(ours, *data, regression)
    s2, t2 = score(theirs, *data, regression)
    print(f"{name:28s} {s1:8.4f} {s2:9.4f}   ({t1:5.2f}s vs {t2:5.2f}s)")


def main(args):
    D, n, s = args.max_depth, args.n_estimators, args.seed

    print("Classification: test accuracy  ours   sklearn")
    row("Single tree", tree.DecisionTreeClassifier("entropy", max_depth=D),
        sk_tree.DecisionTreeClassifier(criterion="entropy", max_depth=D, random_state=s),
        args, False)
    row(f"Bagging ({n} trees)", RandomForest("entropy", n, D, seed=s),
        RandomForestClassifier(n, criterion="entropy", max_depth=D, max_features=None,
                               random_state=s), args, False)
    row(f"Random forest ({n} trees)", RandomForest("entropy", n, D, seed=s, max_features="sqrt"),
        RandomForestClassifier(n, criterion="entropy", max_depth=D, max_features="sqrt",
                               random_state=s), args, False)

    print("\nRegression: test RMSE          ours   sklearn")
    row("Single tree", tree.DecisionTreeRegressor(max_depth=D),
        sk_tree.DecisionTreeRegressor(max_depth=D, random_state=s), args, True)
    row(f"Bagging ({n} trees)", RandomForest(n_estimators=n, max_depth=D, seed=s, is_regression=True),
        RandomForestRegressor(n, max_depth=D, max_features=None, random_state=s), args, True)
    row(f"Random forest ({n} trees)",
        RandomForest(n_estimators=n, max_depth=D, seed=s, is_regression=True, max_features=0.33),
        RandomForestRegressor(n, max_depth=D, max_features=0.33, random_state=s), args, True)

    d, m = args.boost_depth, args.boost_estimators
    row(f"AdaBoost.R2 ({m} trees, d={d})",
        AdaptiveBoostingEnsemble(max_depth=d, n_estimators=m, seed=s),
        AdaBoostRegressor(sk_tree.DecisionTreeRegressor(max_depth=d), n_estimators=m,
                          loss="linear", random_state=s), args, True)
    row(f"Gradient boosting ({m}, d={d})",
        GradientBoostingEnsemble(max_depth=d, n_estimators=m, learning_rate=0.2),
        GradientBoostingRegressor(n_estimators=m, max_depth=d, learning_rate=0.2,
                                  criterion="squared_error",
                                  random_state=s), args, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--car_path", default="data/car.csv")
    parser.add_argument("--house_path", default="data/house.csv")
    parser.add_argument("--synthetic", action="store_true",
                        help="use generated data instead of car.csv / house.csv")
    parser.add_argument("--n_estimators", type=int, default=21)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--boost_estimators", type=int, default=20)
    parser.add_argument("--boost_depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    main(parser.parse_args())
