"""06: AdaBoost for regression (AdaBoost.R2).

What is NEW compared to 05
- Weights are updated multiplicatively and in a principled way: after each
  tree, the weight of a point that the tree predicted well is multiplied by
  beta < 1, so the next training set (resampled with these weights) is
  dominated by the hard points.
- Trees are no longer equal: each gets an importance log(1 / beta) that grows
  with its accuracy, and a tree whose average loss is >= 0.5 stops the
  boosting.
- Trees are combined by the importance-weighted median of their
  predictions, which is robust to a few badly wrong trees.
See the docstring of AdaptiveBoostingEnsemble in model.py for the algorithm.

Run: python 06_adaptive_boosting.py --max_depth 3 --n_estimators 20 --loss linear
"""
import importlib

import numpy as np

from model import AdaptiveBoostingEnsemble as BoostingModel

boosting = importlib.import_module("05_boosting")


if __name__ == "__main__":
    parser = boosting.get_parser(__doc__.splitlines()[0])
    parser.add_argument("--loss", choices=["linear", "square", "exponential"],
                        default="linear", help="how a point's error is turned into a loss in [0, 1]")
    args = parser.parse_args()
    np.random.seed(args.seed)
    model = BoostingModel(max_depth=args.max_depth, n_estimators=args.n_estimators,
                          loss=args.loss, seed=args.seed)
    boosting.compare(model, args, "AdaBoost.R2")
