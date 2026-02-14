import numpy as np

def RegularizationL2(w, lam, lam2):
    return (lam / 2) * np.sum(w ** 2)

def RegularizationL1(w, lam, lam2):
    return lam * np.sum(np.abs(w))

def ElasticNet(w, lam1, lam2):
    return lam1 * np.sum(np.abs(w)) + (lam2 / 2) * np.sum(w ** 2)

class LinearRegression:
    def __init__(self, lr=0.01, epochs=2000, lam=0.1 , lam2=0.0, regulization="L2"):
        self.lr = lr
        self.epochs = epochs
        self.lam = lam
        self.lam2 = lam2
        self.regulization = regulization
        if regulization == "L2":
            self.reg_func = RegularizationL2
        elif regulization == "L1":
            self.reg_func = RegularizationL1
        elif regulization == "ElasticNet":
            self.reg_func = ElasticNet

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        m, k = X.shape
        self.w = np.zeros(k)
        self.b = 0.0

        for _ in range(self.epochs):
            y_hat = X @ self.w + self.b
            err = y_hat - y

            dw = (1/m) * (X.T @ err) + self.reg_func(self.w, self.lam, self.lam2)
            db = (1/m) * np.sum(err)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        X = np.array(X, dtype=float)
        return X @ self.w + self.b

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression(lr=0.1, epochs=2000, lam=0.5, lam2=0.1, regulization="ElasticNet")
model.fit(X, y)

print("Weights:", model.w)
print("Bias:", model.b)
print("Predictions:", model.predict(X))
