import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self, degree: int):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.weights = None

    def _poly_features(self, x: np.ndarray) -> np.ndarray:

        x = x.reshape(-1)
        X = np.ones((x.shape[0], self.degree + 1), dtype=float)
        for d in range(1, self.degree + 1):
            X[:, d] = x ** d
        return X

    def fit(self, x, y):
        x = np.array(x, dtype=float).reshape(-1)
        y = np.array(y, dtype=float).reshape(-1, 1)

        X = self._poly_features(x)

        self.weights = np.linalg.pinv(X) @ y
        return self

    def predict(self, x):
        if self.weights is None:
            raise RuntimeError("Model not fitted yet.")
        x = np.array(x, dtype=float).reshape(-1)
        X = self._poly_features(x)
        return (X @ self.weights).reshape(-1)

def sklearn_comparison(x, y, degree):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x.reshape(-1, 1), y)

    coefs = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_
    print("Sklearn Weights (b0, b1, b2):", [intercept] + list(coefs))
    print("Sklearn Prediction for [6]:", model.predict([[6]]).flatten()[0])

x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
y = np.array([1, 3, 7, 13, 21, 31], dtype=float)

model = PolynomialRegression(degree=2)
model.fit(x, y)

print("Weights [b0, b1, b2]:", model.weights.reshape(-1))
print("Prediction for [6]:", model.predict([6]))

sklearn_comparison(x, y, degree=2)