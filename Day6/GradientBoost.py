import numpy as np
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

class GradientBoostingWithSklearnTrees:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)

        self.init_ = None
        self.trees_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n = X.shape[0]
        if n != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self.init_ = float(np.mean(y))
        self.trees_ = []

        pred = np.full(n, self.init_, dtype=float)

        for m in range(self.n_estimators):
            residual = y - pred

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + m
            )
            tree.fit(X, residual)

            pred += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)

        return self

    def predict(self, X):
        if self.init_ is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.init_, dtype=float)

        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X)

        return pred

def xg_boost(X, y, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    model = XGBRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        random_state=int(random_state),
        verbosity=0
    )
    model.fit(X, y)
    return model.predict(X)

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    X = rng.normal(size=(250, 2))
    y = 2.5 * X[:, 0] - 1.7 * X[:, 1] + 0.4 * np.sin(4 * X[:, 0]) + rng.normal(0, 0.25, size=250)

    model = GradientBoostingWithSklearnTrees(n_estimators=120, learning_rate=0.08, max_depth=3)
    model.fit(X, y)

    preds = model.predict(X)
    mse = np.mean((y - preds) ** 2)
    print("MSE:", float(mse))
    print("First 5 preds:", preds[:5])

    print("\nXGBoost Predictions:")
    xgb_preds = xg_boost(X, y, n_estimators=120, learning_rate=0.08, max_depth=3)
    print("First 5 XGBoost preds:", xgb_preds[:5])