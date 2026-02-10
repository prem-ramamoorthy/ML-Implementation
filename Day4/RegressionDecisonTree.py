import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

X, y_class = make_moons(n_samples=500, noise=0.25, random_state=42)

y = (X[:, 0]**2 + X[:, 1]**2) + np.random.normal(0, 0.05, size=len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def sklearn_predict():
    reg = DecisionTreeRegressor(
        max_depth=6,          # pre-pruning
        min_samples_leaf=5,   # pre-pruning
        ccp_alpha=0.001,      # post-pruning
        random_state=42
    )

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    
    print("Sklearn Predictions:", y_pred)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2 Score:", r2)
    
class DecisionTreeRegressorScratch:

    class Node:
        __slots__ = ("feature", "threshold", "left", "right", "value")
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

        def is_leaf(self):
            return self.value is not None

    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.root = None

    @staticmethod
    def _sse(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        mu = y.mean()
        diff = y - mu
        return float(np.dot(diff, diff))

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        parent_sse = self._sse(y)

        best_feature = None
        best_threshold = None
        best_gain = 0.0

        if np.allclose(y, y[0]):
            return None, None, 0.0

        for f in range(n_features):
            x = X[:, f]

            uniq = np.unique(x)
            if uniq.size <= 1:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            for t in thresholds:
                left_mask = x <= t
                right_mask = ~left_mask

                n_left = int(left_mask.sum())
                n_right = n_samples - n_left

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                sse_left = self._sse(y[left_mask])
                sse_right = self._sse(y[right_mask])
                child_sse = sse_left + sse_right

                gain = parent_sse - child_sse
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = float(t)

        return best_feature, best_threshold, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int):
        n_samples = X.shape[0]
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.allclose(y, y[0])
        ):
            return self.Node(value=float(y.mean()))

        feat, thr, gain = self._best_split(X, y)
        if feat is None or gain <= 1e-12:
            return self.Node(value=float(y.mean()))

        mask = X[:, feat] <= thr
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return self.Node(feature=feat, threshold=thr, left=left, right=right)

    def fit(self, X_train, y_train):
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X_train must be 2D: (n_samples, n_features)")
        if y.shape[0] != X.shape[0]:
            raise ValueError("y_train length must match X_train rows")
        self.root = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x: np.ndarray) -> float:
        node = self.root
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X_test):
        if self.root is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X_test, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._predict_one(row) for row in X], dtype=float)


def decision_tree_regression_from_scratch(X_train, y_train, X_test,
                                         max_depth=5, min_samples_split=2, min_samples_leaf=1):
    model = DecisionTreeRegressorScratch(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)

sklearn_predict()

preds = decision_tree_regression_from_scratch(
        X_train, y_train, X_test,
        max_depth=3, min_samples_split=2, min_samples_leaf=1
    )
print("Predictions:", preds)