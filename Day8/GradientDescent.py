import numpy as np

def gradient_descent_linear(X, y, lr=0.01, epochs=2000):
    m, k = X.shape
    w = np.zeros(k)
    b = 0.0

    for _ in range(epochs):
        y_hat = X @ w + b
        err = y_hat - y

        dw = (1/m) * (X.T @ err)
        db = (1/m) * np.sum(err)

        w -= lr * dw
        b -= lr * db

    return w, b

def stochastic_gradient_descent_linear(X, y, lr=0.01, epochs=2000):
    m, k = X.shape
    w = np.zeros(k)
    b = 0.0

    for _ in range(epochs):
        for i in range(m):
            xi = X[i]
            yi = y[i]

            y_hat = xi @ w + b
            err = y_hat - yi

            dw = err * xi
            db = err

            w -= lr * dw
            b -= lr * db

    return w, b

def mini_batch_gradient_descent_linear(X, y, lr=0.01, epochs=2000, batch_size=32):
    m, k = X.shape
    w = np.zeros(k)
    b = 0.0

    for _ in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            y_hat = X_batch @ w + b
            err = y_hat - y_batch

            dw = (1/len(X_batch)) * (X_batch.T @ err)
            db = (1/len(X_batch)) * np.sum(err)

            w -= lr * dw
            b -= lr * db

    return w, b

X = np.array([[1.0], [2.0], [3.0]])
y = np.array([2.0, 4.0, 6.0])

w, b = gradient_descent_linear(X, y, lr=0.1, epochs=20)
print("w:", w, "b:", b)

w, b = stochastic_gradient_descent_linear(X, y, lr=0.1, epochs=20)
print("w:", w, "b:", b)

w, b = mini_batch_gradient_descent_linear(X, y, lr=0.1, epochs=20)
print("w:", w, "b:", b)
