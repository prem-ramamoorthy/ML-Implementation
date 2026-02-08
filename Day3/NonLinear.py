import numpy as np

def rbf_kernel(X1, X2, gamma):
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    s1 = np.sum(X1**2, axis=1).reshape(-1, 1)
    s2 = np.sum(X2**2, axis=1).reshape(1, -1)
    return np.exp(-gamma * (s1 + s2 - 2.0 * (X1 @ X2.T))) # RBF kernel looks like this: K(x, y) = exp(-gamma * ||x - y||^2) (which can be expanded to the above form using the identity ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y)

class SVMRBF:
    def __init__(self, C=1.0, gamma=0.5, tol=1e-3, eps=1e-3, max_passes=5, max_iters=20000, seed=0):
        self.C = float(C)
        self.gamma = float(gamma)
        self.tol = float(tol)
        self.eps = float(eps)
        self.max_passes = int(max_passes)
        self.max_iters = int(max_iters)
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        y = np.where(y > 0, 1.0, -1.0)
        n = X.shape[0]

        K = rbf_kernel(X, X, self.gamma)
        a = np.zeros(n, dtype=float)
        b = 0.0 

        def f(i):
            return (a * y) @ K[:, i] + b

        passes = 0
        iters = 0

        while passes < self.max_passes and iters < self.max_iters:
            num_changed = 0
            for i in range(n):
                Ei = f(i) - y[i]
                if ((y[i] * Ei < -self.tol and a[i] < self.C - 1e-12) or
                    (y[i] * Ei >  self.tol and a[i] > 1e-12)):

                    j = i
                    while j == i:
                        j = int(self.rng.integers(0, n))

                    Ej = f(j) - y[j]
                    ai_old, aj_old = a[i], a[j]

                    if y[i] != y[j]:
                        L = max(0.0, aj_old - ai_old)
                        H = min(self.C, self.C + aj_old - ai_old)
                    else:
                        L = max(0.0, ai_old + aj_old - self.C)
                        H = min(self.C, ai_old + aj_old)

                    if H - L < 1e-12:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    aj_new = aj_old - (y[j] * (Ei - Ej)) / eta
                    if aj_new > H: aj_new = H
                    elif aj_new < L: aj_new = L

                    if abs(aj_new - aj_old) < self.eps:
                        continue

                    ai_new = ai_old + y[i] * y[j] * (aj_old - aj_new)

                    b1 = b - Ei - y[i] * (ai_new - ai_old) * K[i, i] - y[j] * (aj_new - aj_old) * K[i, j]
                    b2 = b - Ej - y[i] * (ai_new - ai_old) * K[i, j] - y[j] * (aj_new - aj_old) * K[j, j]

                    if 0.0 < ai_new < self.C:
                        b = b1
                    elif 0.0 < aj_new < self.C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)

                    a[i], a[j] = ai_new, aj_new
                    num_changed += 1

            passes = passes + 1 if num_changed == 0 else 0
            iters += 1

        self.X_ = X
        self.y_ = y
        self.a_ = a
        self.b_ = b
        sv = a > 1e-10
        self.sv_X_ = X[sv]
        self.sv_y_ = y[sv]
        self.sv_a_ = a[sv]
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        K = rbf_kernel(self.sv_X_, X, self.gamma)
        return (self.sv_a_ * self.sv_y_) @ K + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1.0, -1.0)

def scipy_svm(X, y, C=1.0, gamma=0.5):
    from sklearn.svm import SVC
    model = SVC(kernel='poly', C=C, gamma=gamma)
    model.fit(X, y)
    return model

if __name__ == "__main__":
    X = np.array([
        [0.0, 0.0], [0.2, 0.1], [0.1, 0.2],
        [1.0, 1.0], [1.1, 0.9], [0.9, 1.1]
    ])
    y = np.array([-1, -1, -1, 1, 1, 1])

    model = SVMRBF(C=10.0, gamma=5.0, max_passes=10, seed=42).fit(X, y)
    print("train preds:", model.predict(X))
    print("decision:", model.decision_function(X))
    sklearn_model = scipy_svm(X, y, C=10.0, gamma=5.0)
    print("sklearn train preds:", sklearn_model.predict(X))