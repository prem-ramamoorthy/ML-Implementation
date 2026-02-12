import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def fit(self, X, y, w):
        n_samples, n_features = X.shape
        min_error = float("inf")

        for j in range(n_features):
            xj = X[:, j]
            thresholds = np.unique(xj)

            for thr in thresholds:
                pred = np.ones(n_samples)
                pred[xj < thr] = -1

                err = np.sum(w[pred != y])
                if err > 0.5:
                    err = 1.0 - err
                    pol = -1
                else:
                    pol = 1

                if err < min_error:
                    min_error = err
                    self.feature_index = j
                    self.threshold = thr
                    self.polarity = pol

        return min_error

    def predict(self, X):
        xj = X[:, self.feature_index]
        pred = np.ones(X.shape[0])
        pred[xj < self.threshold] = -1
        return self.polarity * pred


class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        y = np.where(y <= 0, -1.0, 1.0)

        n_samples = X.shape[0]
        w = np.full(n_samples, 1.0 / n_samples)

        self.stumps = []
        self.alphas = []

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            err = stump.fit(X, y, w)

            err = np.clip(err, 1e-12, 1 - 1e-12)
            alpha = 0.5 * np.log((1 - err) / err)

            pred = stump.predict(X)
            w *= np.exp(-alpha * y * pred)
            w /= np.sum(w)

            self.stumps.append(stump)
            self.alphas.append(alpha)

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        F = np.zeros(X.shape[0], dtype=float)
        for alpha, stump in zip(self.alphas, self.stumps):
            F += alpha * stump.predict(X)
        return F

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def predict_proba(self, X):
        F = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-2.0 * F))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

def scikit_learn_comparison(x_train=None, y_train=None, x_test=None, y_test=None):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import accuracy_score

    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=20,
        learning_rate=0.5,
        random_state=42
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    print("Scikit-learn AdaBoost Accuracy:", accuracy_score(y_test, preds))

if __name__ == "__main__":
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [1.5, 1.8],
        [3.0, 3.2],
        [3.5, 2.8],
        [2.8, 3.0],
    ])
    y = np.array([-1, -1, -1, 1, 1, 1])

    model = AdaBoost(n_estimators=20).fit(X, y)
    preds = model.predict(X)
    probs = model.predict_proba(X)

    print("preds:", preds)
    print("probs:", np.round(probs, 3))

    scikit_learn_comparison(X, y, X, y)
