import numpy as np

class MultiLinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)

        n_samples = X.shape[0]

        X_bias = np.hstack((np.ones((n_samples, 1)), X)) # Add bias term (intercept) to the features
        self.weights = np.linalg.inv(X_bias.T @ X_bias) @ (X_bias.T @ y)

    def predict(self, X):
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]

        X_bias = np.hstack((np.ones((n_samples, 1)), X))

        return (X_bias @ self.weights).flatten()

def sklearn_comparison(X, y):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X, y)

    print("Sklearn Weights (b0, b1, b2):", [model.intercept_] + list(model.coef_))

X = [
    [1, 0],
    [0, 1],
    [1, 1],
    [2, 1],
    [3, 2]
]

y = [5, 6, 9, 12, 20]

model = MultiLinearRegression()
model.fit(X, y)

print("Weights (b0, b1, b2):", model.weights.flatten())

pred = model.predict([[2, 2]])
print("Prediction for [2,2]:", pred)

sklearn_comparison(X, y)