"""07: gradient boosting for regression.

What is NEW compared to 05-06
- No sample weights at all: each new tree is trained on the residuals
  y - F(x) of the current ensemble F, i.e. it learns what is still missing.
- The residual is the negative gradient of the squared loss
  1/2 (y - F(x))^2 with respect to the prediction F(x), so adding a tree
  fitted to it is a gradient-descent step in function space. Swapping in
  another loss (and its gradient) gives gradient boosting for that loss.
- learning_rate shrinks every step: smaller steps need more trees but
  usually generalise better.
- We also print the RMSE as trees are added: the train error keeps falling,
  while the test error eventually flattens or rises (overfitting).

Run: python 07_gradient_boosting.py --max_depth 3 --n_estimators 20 --learning_rate 0.2
"""
import importlib

import numpy as np

from utils import get_data, rmse
from model import GradientBoostingEnsemble

boosting = importlib.import_module("05_boosting")


if __name__ == "__main__":
    parser = boosting.get_parser(__doc__.splitlines()[0])
    parser.add_argument("--learning_rate", type=float, default=0.2)
    args = parser.parse_args()
    np.random.seed(args.seed)
    model = GradientBoostingEnsemble(max_depth=args.max_depth, n_estimators=args.n_estimators,
                                     learning_rate=args.learning_rate)
    boosting.compare(model, args, "Gradient boosting")

    # RMSE after each stage (the model is already fitted by compare)
    X_train, X_test, y_train, y_test = get_data(args, "reg")
    print("trees | train RMSE | test RMSE")
    for m in range(0, len(model.models) + 1, max(1, len(model.models) // 10)):
        print(f"{m:5d} | {rmse(model.predict(X_train, m), y_train):10.4f} | "
              f"{rmse(model.predict(X_test, m), y_test):9.4f}")
