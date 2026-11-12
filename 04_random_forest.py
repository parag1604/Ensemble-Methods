"""04: random forests = bagging + random feature subsets.

What is NEW compared to 03
- In bagging all trees still see all features, so if one feature is very
  strong every tree splits on it first and the trees stay correlated (02
  showed why correlated voters help less).
- A random forest additionally lets each split choose only among a random
  subset of the features (max_features), e.g. sqrt(D) features for
  classification or about D/3 for regression. Individual trees get slightly
  worse, but they become more diverse, and the average gets better.
- Try --max_features 1.0 (all features): that is exactly bagging again.

Run: python 04_random_forest.py --task clf
     python 04_random_forest.py --task reg
"""
import importlib

# reuse the experiment from 03 (module names starting with a digit need importlib)
bagging = importlib.import_module("03_bagging")


def parse_max_features(value: str):
    """'sqrt' / 'log2' stay strings; numbers become a fraction of the features."""
    return value if value in ("sqrt", "log2") else float(value)


if __name__ == "__main__":
    parser = bagging.get_parser(__doc__.splitlines()[0])
    parser.add_argument("--max_features", type=parse_max_features, default=None,
                        help="features tried per split: sqrt, log2 or a fraction "
                             "(default: sqrt for clf, 0.33 for reg)")
    args = parser.parse_args()
    if args.max_features is None:
        args.max_features = "sqrt" if args.task == "clf" else 0.33
    bagging.main(args, max_features=args.max_features, name="Random forest")
