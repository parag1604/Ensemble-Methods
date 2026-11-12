"""01: how much of the data does a bootstrap sample contain?

What this file teaches
- Bagging trains every model on a bootstrap sample: N points drawn WITH
  replacement from the N training points. Some points are drawn several
  times, others not at all.
- A given point is missed by all N draws with probability (1 - 1/N)^N, which
  tends to 1/e = 0.368 as N grows. So each bootstrap sample contains only
  about 1 - 1/e = 63.2% of the distinct points. This is what makes the
  models in a bagging ensemble differ from each other, and the ~36.8% of
  points a model never saw ("out-of-bag") can serve as its validation set.

Run: python 01_bootstrap_unique_points.py --n 1000000
"""
import argparse

import numpy as np


def main(n: int) -> None:
    idxs = np.random.choice(n, n, replace=True)
    unique_data_size = len(np.unique(idxs))
    unique_data_percentage = unique_data_size / n * 100
    print(f"Unique data points: {unique_data_size} ({unique_data_percentage:.2f}%)")
    print(f"Exact expectation 1 - (1 - 1/n)^n: {(1 - (1 - 1 / n) ** n) * 100:.2f}%")
    print(f"Limit 1 - 1/e:                     {(1 - 1 / np.e) * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1_000_000, help="dataset size")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    np.random.seed(args.seed)
    main(args.n)
