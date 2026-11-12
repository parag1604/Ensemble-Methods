"""Ensembles of decision trees: bagging / random forests and three kinds of boosting.

The base learners are scikit-learn's decision trees (the from-scratch versions
are in the Decision-Trees repository); here the focus is on how the trees are
trained and combined.
"""
from typing import Optional, Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForest:
    """Bagging (max_features=None) or a random forest (max_features < all).

    Each tree is trained on a bootstrap sample (N points drawn with replacement).
    Classification predictions are combined by majority vote, regression
    predictions by averaging. With max_features set, every split of every tree
    only looks at a random subset of the features, which decorrelates the trees.
    """

    def __init__(
            self,
            info_criteria: str = "entropy",
            n_estimators: int = 10,
            max_depth: Optional[int] = 5,
            seed: int = 2024,
            is_regression: bool = False,
            max_features: Union[None, str, int, float] = None,
    ) -> None:
        self.info_criteria = info_criteria
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.rng = np.random.RandomState(seed)
        self.is_regression = is_regression
        self.max_features = max_features
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.trees = []
        for _ in range(self.n_estimators):
            seed = self.rng.randint(2 ** 31 - 1)  # the trees' own feature sampling
            if not self.is_regression:
                tree = DecisionTreeClassifier(
                    criterion=self.info_criteria,
                    max_depth=self.max_depth,
                    max_features=self.max_features,
                    random_state=seed,
                )
            else:
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    max_features=self.max_features,
                    random_state=seed,
                )
            # bootstrap the data
            idxs = self.rng.choice(len(X), len(X), replace=True)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_preds = np.array([tree.predict(X) for tree in self.trees])  # (n_trees, N)
        if self.is_regression:
            return y_preds.mean(axis=0)
        # majority vote per sample
        classes, votes = np.unique(y_preds, return_inverse=True)
        votes = votes.reshape(y_preds.shape)
        counts = np.array([np.bincount(votes[:, i], minlength=len(classes))
                           for i in range(votes.shape[1])])
        return classes[np.argmax(counts, axis=1)]


class BoostingEnsemble:
    """A first, intuitive take on boosting (regression).

    Train a tree, look at which points the ensemble still gets wrong, and train
    the next tree with those points weighted more (weight proportional to the
    squared error). The prediction is the plain average of all trees.
    This heuristic is NOT a standard algorithm; AdaBoost (below) turns the same
    idea into a principled weighting of points and of trees.
    """

    def __init__(
            self,
            max_depth: int,
            n_estimators: int,
    ) -> None:
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.models = []
        model = DecisionTreeRegressor(max_depth=self.max_depth)
        self.models.append(model)
        model.fit(X, y)
        for _ in range(self.n_estimators - 1):
            preds = self.predict(X)
            errors = (y - preds) ** 2
            errors_sum = errors.sum()
            if errors_sum < 0.01:  # (almost) perfect fit: nothing left to fix
                break
            errors = errors / errors_sum  # sample weights that sum to 1
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            self.models.append(model)
            model.fit(X, y, sample_weight=errors)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([model.predict(X) for model in self.models], axis=0)


class AdaptiveBoostingEnsemble:
    """AdaBoost for regression (AdaBoost.R2, Drucker 1997).

    Keeps a weight per training point. Each round:
      1. resample the training set according to the weights and fit a tree;
      2. compute each point's loss L_i in [0, 1] (error / largest error);
      3. the weighted average loss Lbar gives the tree a confidence
         beta = Lbar / (1 - Lbar) and an importance log(1 / beta)
         (accurate trees get large importance; stop if Lbar >= 0.5);
      4. shrink the weights of well-predicted points: w_i <- w_i * beta^(1 - L_i).
    The prediction is the importance-weighted MEDIAN of the trees' predictions.
    """

    def __init__(
            self,
            max_depth: int,
            n_estimators: int,
            loss: str = "linear",
            seed: int = 2024,
    ) -> None:
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.loss = loss
        self.rng = np.random.RandomState(seed)
        self.models = []
        self.model_importances = []

    def _point_losses(self, y: np.ndarray, preds: np.ndarray) -> np.ndarray:
        errors = np.abs(y - preds)
        errors = errors / errors.max() if errors.max() > 0 else errors
        if self.loss == "square":
            return errors ** 2
        if self.loss == "exponential":
            return 1 - np.exp(-errors)
        return errors  # linear

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.models, self.model_importances = [], []
        weights = np.ones(len(X)) / len(X)
        for _ in range(self.n_estimators):
            # create and fit new model after resampling according to the weights
            idxs = self.rng.choice(len(X), size=len(X), p=weights)
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X[idxs], y[idxs])
            # per-point losses in [0, 1] and the weighted average loss
            losses = self._point_losses(y, model.predict(X))
            avg_loss = np.sum(weights * losses)
            if avg_loss >= 0.5 and self.models:
                break  # this tree is worse than useless; stop boosting
            avg_loss = max(avg_loss, 1e-10)  # a perfect tree: avoid log(1/0)
            beta = avg_loss / (1 - avg_loss)
            # calculate model importance
            self.models.append(model)
            self.model_importances.append(np.log(1 / beta))
            # well-predicted points (small loss) get their weight shrunk the most
            weights = weights * beta ** (1 - losses)
            weights = weights / weights.sum()

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([model.predict(X) for model in self.models]).T  # (N, M)
        importances = np.array(self.model_importances)
        # weighted median: sort each row, pick the first prediction at which the
        # cumulative importance reaches half of the total
        order = np.argsort(preds, axis=1)
        cum = np.cumsum(importances[order], axis=1)
        median_pos = np.argmax(cum >= 0.5 * cum[:, -1:], axis=1)
        median_idx = order[np.arange(len(X)), median_pos]
        return preds[np.arange(len(X)), median_idx]


class GradientBoostingEnsemble:
    """Gradient boosting for regression with the squared loss.

    Start from the constant prediction mean(y). Each new tree is fit to the
    current residuals y - F(x), which are the negative gradient of the loss
    1/2 (y - F(x))^2 with respect to F(x), and is added with a small step
    (learning_rate): F(x) <- F(x) + learning_rate * tree(x).
    """

    def __init__(
            self,
            max_depth: int,
            n_estimators: int,
            learning_rate: float = 0.1,
    ) -> None:
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.mean = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.models = []
        self.mean = y.mean()
        residue = y - self.mean
        for _ in range(self.n_estimators):
            if np.abs(residue).sum() < 0.01:
                break
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            self.models.append(model)
            model.fit(X, residue)
            residue = residue - self.learning_rate * model.predict(X)

    def predict(self, X: np.ndarray, n_models: Optional[int] = None) -> np.ndarray:
        """Prediction using the first n_models trees (all trees by default)."""
        y_pred = np.ones(len(X)) * self.mean
        for model in self.models[:n_models]:
            y_pred += self.learning_rate * model.predict(X)
        return y_pred
