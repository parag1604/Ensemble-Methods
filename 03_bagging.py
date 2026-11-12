"""03: bagging (bootstrap aggregating) of decision trees.

What is NEW compared to 01-02
- Train n_estimators deep trees, each on its own bootstrap sample (01), and
  combine them: majority vote for classification (02), average for regression.
- A single deep tree has low bias but high variance (it changes a lot if the
  training set changes). Averaging many such trees keeps the low bias and
  reduces the variance, so the test score usually improves over a single tree
  of the same depth.

Run: python 03_bagging.py --task clf     (car.csv)
     python 03_bagging.py --task reg     (house.csv)
"""
import argparse

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from utils import load_data, standardize_targets, report_classification, report_regression
from model import RandomForest as RFModel


def main(args, max_features=None, name="Bagging") -> None:
    if args.task == "clf":
        X_train, X_test, y_train, y_test = load_data(args.car_path)
        single = DecisionTreeClassifier(criterion=args.criteria, max_depth=args.max_depth,
                                        random_state=args.seed)
        ensemble = RFModel(args.criteria, args.n_estimators, args.max_depth,
                           seed=args.seed, max_features=max_features)
        report = report_classification
    else:
        X_train, X_test, y_train, y_test = load_data(args.house_path, is_regression=True)
        y_train, y_test = standardize_targets(y_train, y_test)
        single = DecisionTreeRegressor(max_depth=args.max_depth, random_state=args.seed)
        ensemble = RFModel(n_estimators=args.n_estimators, max_depth=args.max_depth,
                           seed=args.seed, is_regression=True, max_features=max_features)
        report = report_regression
    report("Single tree", single, X_train, X_test, y_train, y_test)
    report(f"{name} ({args.n_estimators} trees)", ensemble, X_train, X_test, y_train, y_test)


def get_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--task", choices=["clf", "reg"], default="clf")
    parser.add_argument("--car_path", default="data/car.csv")
    parser.add_argument("--house_path", default="data/house.csv")
    parser.add_argument("--criteria", choices=["entropy", "gini"], default="entropy")
    parser.add_argument("--n_estimators", type=int, default=21)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2024)
    return parser


if __name__ == "__main__":
    main(get_parser(__doc__.splitlines()[0]).parse_args())
