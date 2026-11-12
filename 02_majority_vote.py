"""02: why a committee of weak learners can be strong (and when it cannot).

What is NEW compared to 01
- Suppose each of n classifiers is wrong with probability c = 1/2 - eps,
  i.e. only slightly better than a coin flip, and suppose their mistakes are
  INDEPENDENT. A majority vote is wrong only if more than half of them are
  wrong, which is a binomial tail:
      P(majority wrong) = sum_{k > n/2} C(n, k) c^k (1 - c)^(n - k)
  (we use odd n so that there are no ties).
- The plot shows that the accuracy of the vote climbs towards 1 as n grows,
  slowly for tiny eps and quickly for larger eps.
- The catch: the models in a real ensemble are trained on the same data, so
  their mistakes are correlated and the gain is smaller. Bagging (03) and
  random forests (04) are tricks to make the trees as independent as possible.

Run: python 02_majority_vote.py
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


def prob_majority_correct(n: int, c: float) -> float:
    # the vote is wrong when at least (n + 1) / 2 of the n learners are wrong
    return 1 - binom.sf(n // 2, n, c)  # sf(k) = P(K > k)


def main(epsilons, max_n: int) -> None:
    xs = np.arange(1, max_n + 1, 2)  # odd committee sizes
    for eps in epsilons:
        c = 1 / 2 - eps
        prob_corrects = [prob_majority_correct(n, c) for n in xs]
        plt.plot(xs, prob_corrects, label=f"each learner {100 * (1 - c):.0f}% accurate")
        print(f"eps={eps}: single learner {1 - c:.2f} -> "
              f"majority of {xs[-1]}: {prob_corrects[-1]:.3f}")
    plt.grid()
    plt.xlabel("Number of weak learners (independent errors)")
    plt.ylabel("Probability of correct classification")
    plt.xlim(1, xs[-1])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--max_n", type=int, default=201)
    args = parser.parse_args()
    main(args.eps, args.max_n)
