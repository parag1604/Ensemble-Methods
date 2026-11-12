"""05: boosting, the intuition: train the next tree on the current mistakes.

What is NEW compared to 03-04
- Bagging trains its trees independently and in parallel. Boosting trains
  them one after the other, and each new tree focuses on the points the
  ensemble still gets wrong.
- Here we do this in the most direct way: weight each training point by its
  current squared error and fit the next (shallow) tree with those sample
  weights. The prediction is the plain average of all trees.
- This is a heuristic, not a standard algorithm: averaging lets the early
  trees' mistakes persist, so the gain over a single tree is limited.
  06 (AdaBoost) and 07 (gradient boosting) make the idea principled.

Run: python 05_boosting.py --max_depth 3 --n_estimators 20   (house.csv)
     add --synthetic to use generated data instead of the CSV file
"""
import argparse

import numpy as np
from tree import DecisionTreeRegressor
from utils import get_data, report_regression
from model import BoostingEnsemble as BoostingModel


def get_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_path", default="data/house.csv")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--n_estimators", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--synthetic", action="store_true",
                        help="use generated data instead of house.csv")
    return parser


def compare(model, args, name: str) -> None:
    """Single tree of the same depth vs. the boosted ensemble (standardized prices)."""
    X_train, X_test, y_train, y_test = get_data(args, "reg")
    report_regression("Single tree", DecisionTreeRegressor(max_depth=args.max_depth),
                      X_train, X_test, y_train, y_test)
    report_regression(name, model, X_train, X_test, y_train, y_test)
    print(f"Number of trees used: {len(model.models)}")


if __name__ == "__main__":
    args = get_parser(__doc__.splitlines()[0]).parse_args()
    np.random.seed(args.seed)
    compare(BoostingModel(max_depth=args.max_depth, n_estimators=args.n_estimators),
            args, "Boosting (heuristic)")
