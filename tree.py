"""Decision trees (CART) from scratch, used as the base learners of every ensemble.

Adapted from model.py of the companion repository Decision-Trees
(https://github.com/parag1604/Decision-Trees), which grows the same trees one
threshold at a time for readability. This version adds what the ensembles need:

- max_depth          : limit on the number of questions from root to leaf
                       (None or a negative number: unlimited);
- min_samples_split  : a node with fewer training points becomes a leaf;
- max_features       : each split looks only at a random subset of the
                       features (None/all, "sqrt", "log2", an int, or a
                       fraction in (0, 1]); this is what makes a random forest;
- sample_weight      : weighted impurities and weighted leaf values, used by
                       the boosting heuristic in 05;
- a vectorised split search: for every feature the points are sorted once and
  the impurities of ALL thresholds come from cumulative sums, instead of one
  pass over the data per threshold. Same splits, much faster.

As in Decision-Trees, a split asks "x[j] < threshold?", thresholds are midpoints
between consecutive distinct values, and ties between equally good splits go to
the first one found.
"""
from typing import Optional, Union

import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, value):
        self.feature = -1        # feature used for the split (-1 for a leaf)
        self.threshold = 0.0     # go left if x[feature] < threshold
        self.left = None
        self.right = None
        self.value = value       # leaf output: class-weight vector or mean target


class _BaseTree:
    is_regression = False

    def __init__(
            self,
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            max_features: Union[None, str, int, float] = None,
            random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None

    # ---- to be provided by the subclasses ----------------------------------
    def _encode_targets(self, y): ...
    def _leaf_value(self, Y, w): ...
    def _node_is_pure(self, Y, w): ...
    def _best_threshold(self, x_sorted, Y_sorted, w_sorted): ...
    def _output(self, values): ...

    # ---- growing the tree --------------------------------------------------
    def _n_features_per_split(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        return max(1, min(int(mf), n_features))

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=float)
        Y = self._encode_targets(np.asarray(y))
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, float)
        self.rng_ = np.random.RandomState(self.random_state)
        self.k_features_ = self._n_features_per_split(X.shape[1])
        depth = None if self.max_depth is None or self.max_depth < 0 else self.max_depth
        self.root = self._grow(X, Y, w, depth)
        return self

    def _grow(self, X, Y, w, depth):
        node = _Node(self._leaf_value(Y, w))
        # stopping rules: depth budget used up, too few points, or a pure node
        if depth == 0 or len(X) < self.min_samples_split or self._node_is_pure(Y, w):
            return node

        best_gain, best_feature, best_threshold = -np.inf, None, None
        n_features = X.shape[1]
        # random forest: visit the features in random order and stop after
        # k_features_ of them (all of them, in order, for a plain tree). Like
        # scikit-learn, keep looking if none of the visited features can split.
        order = (np.arange(n_features) if self.k_features_ == n_features
                 else self.rng_.permutation(n_features))
        visited = 0
        for j in order:
            if visited >= self.k_features_ and best_feature is not None:
                break
            visited += 1
            idx = np.argsort(X[:, j], kind="stable")
            gain, threshold = self._best_threshold(X[idx, j], Y[idx], w[idx])
            if threshold is not None and gain > best_gain:
                best_gain, best_feature, best_threshold = gain, j, threshold
        if best_feature is None:  # every visited feature is constant here
            return node

        node.feature, node.threshold = int(best_feature), float(best_threshold)
        left = X[:, best_feature] < best_threshold
        child_depth = None if depth is None else depth - 1
        node.left = self._grow(X[left], Y[left], w[left], child_depth)
        node.right = self._grow(X[~left], Y[~left], w[~left], child_depth)
        return node

    # ---- prediction --------------------------------------------------------
    def _leaf_values(self, X: np.ndarray) -> np.ndarray:
        """Route all points down the tree at once; return each point's leaf value."""
        out = [None] * len(X)

        def route(node, idx):
            if node.left is None:
                for i in idx:
                    out[i] = node.value
                return
            go_left = X[idx, node.feature] < node.threshold
            route(node.left, idx[go_left])
            route(node.right, idx[~go_left])

        route(self.root, np.arange(len(X)))
        return np.array(out)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.root is not None, "Model not trained yet"
        return self._output(self._leaf_values(np.asarray(X, dtype=float)))


class DecisionTreeClassifier(_BaseTree):
    """Classification tree; impurity = entropy or Gini index of the (weighted)
    class proportions, leaf = (weighted) majority class."""

    def __init__(self, criterion: str = "entropy", **kwargs) -> None:
        super().__init__(**kwargs)
        if criterion not in ("entropy", "gini"):
            raise ValueError(f"Unknown criterion: {criterion}")
        self.criterion = criterion

    def _encode_targets(self, y):
        # one-hot encode, so class counts on either side of every threshold
        # are a cumulative sum away
        self.classes_, codes = np.unique(y, return_inverse=True)
        return np.eye(len(self.classes_))[codes]

    def _impurity(self, counts: np.ndarray) -> np.ndarray:
        """Impurity of each row of (weighted) class counts."""
        total = counts.sum(axis=-1, keepdims=True)
        p = counts / np.maximum(total, 1e-300)
        if self.criterion == "gini":
            return 1 - np.sum(p ** 2, axis=-1)       # G = 1 - sum_k p_k^2
        with np.errstate(divide="ignore", invalid="ignore"):
            plogp = np.where(p > 0, p * np.log(p), 0.0)
        return -np.sum(plogp, axis=-1)               # H = -sum_k p_k log p_k

    def _leaf_value(self, Y, w):
        return w @ Y                                  # weighted class counts

    def _node_is_pure(self, Y, w):
        return np.count_nonzero(w @ Y) <= 1

    def _best_threshold(self, x, Y, w):
        WY = Y * w[:, None]
        total = WY.sum(axis=0)
        left = np.cumsum(WY, axis=0)[:-1]             # counts left of each cut
        right = total - left
        valid = x[1:] > x[:-1]                         # cut only between distinct values
        if not valid.any():
            return -np.inf, None
        W, WL = total.sum(), left.sum(axis=1)
        # gain = impurity(parent) - sum_child (W_child / W) impurity(child)
        gain = self._impurity(total) - (WL * self._impurity(left)
                                        + (W - WL) * self._impurity(right)) / W
        gain = np.where(valid, gain, -np.inf)
        i = int(np.argmax(gain))
        return gain[i], (x[i] + x[i + 1]) / 2

    def _output(self, values):
        return self.classes_[np.argmax(values, axis=1)]


class DecisionTreeRegressor(_BaseTree):
    """Regression tree; impurity = (weighted) variance of the targets, so the
    gain is the reduction in squared error; leaf = (weighted) mean target."""

    is_regression = True

    def _encode_targets(self, y):
        return y.astype(float)

    def _leaf_value(self, y, w):
        return float(np.sum(w * y) / np.sum(w))

    def _node_is_pure(self, y, w):
        return np.ptp(y) == 0

    def _best_threshold(self, x, y, w):
        valid = x[1:] > x[:-1]
        if not valid.any():
            return -np.inf, None
        # weighted sums on the left of every cut; SSE = sum w y^2 - (sum w y)^2 / sum w
        cw, cy, cy2 = np.cumsum(w), np.cumsum(w * y), np.cumsum(w * y * y)
        W, S, S2 = cw[-1], cy[-1], cy2[-1]
        wl, sl, s2l = cw[:-1], cy[:-1], cy2[:-1]
        wr, sr, s2r = W - wl, S - sl, S2 - s2l
        with np.errstate(divide="ignore", invalid="ignore"):
            sse_children = (s2l - sl ** 2 / wl) + (s2r - sr ** 2 / wr)
        gain = ((S2 - S ** 2 / W) - sse_children) / W   # reduction in weighted variance
        gain = np.where(valid & (wl > 0) & (wr > 0), gain, -np.inf)
        i = int(np.argmax(gain))
        if not np.isfinite(gain[i]):
            return -np.inf, None
        return gain[i], (x[i] + x[i + 1]) / 2

    def _output(self, values):
        return values.astype(float)
