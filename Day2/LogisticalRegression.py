import math
from sklearn.linear_model import LogisticRegression

def logistic_regression_gd(
    X,
    y,
    lr=0.5,
    epochs=100,
    threshold=0.5,
    verbose=True,
    print_every=1
):
    m = len(X)
    n = 1 if isinstance(X[0], (int, float)) else len(X[0])
    
    def dot(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    def sigmoid(z):
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    w = [0.0] * n
    b = 0.0

    if verbose:
        print("Initialized:")
        print("w =", w)
        print("b =", b)
        print("-" * 70)

    for epoch in range(1, epochs + 1):
        dw = [0.0] * n
        db = 0.0

        rows = []

        for xi, yi in zip(X, y):
            z = dot(w, xi) + b 
            p = sigmoid(z)
            err = p - yi

            for j in range(n):
                dw[j] += err * xi[j]
            db += err

            if verbose and (epoch % print_every == 0):
                rows.append((xi, yi, z, p, err))

        dw = [g / m for g in dw]
        db /= m

        w = [wj - lr * dwj for wj, dwj in zip(w, dw)]
        b = b - lr * db

        if verbose and (epoch % print_every == 0):
            print(f"\nEpoch {epoch}")
            if n == 1:
                print("x     y     z=wx+b        p=sigmoid(z)   error(p-y)")
                print("-" * 70)
                for (xi, yi, z, p, err) in rows:
                    print(f"{xi[0]:<5.2f} {yi:<5d} {z:<12.6f} {p:<13.6f} {err:<.6f}")
            else:
                print("x                     y     z=w·x+b       p=sigmoid(z)   error(p-y)")
                print("-" * 70)
                for (xi, yi, z, p, err) in rows:
                    print(f"{str([round(v,2) for v in xi]):<21} {yi:<5d} {z:<12.6f} {p:<13.6f} {err:<.6f}")

            print("\nGradients:")
            print("dw =", [round(v, 6) for v in dw])
            print("db =", round(db, 6))
            print("Updated params:")
            print("w  =", [round(v, 6) for v in w])
            print("b  =", round(b, 6))

    if verbose:
        print("\n" + "=" * 70)
        print("FINAL MODEL")
        print("w =", [round(v, 6) for v in w])
        print("b =", round(b, 6))

        print("\nFINAL PREDICTIONS")
        for xi, yi in zip(X, y):
            z = dot(w, xi) + b
            p = sigmoid(z)
            pred = 1 if p >= threshold else 0
            print(f"x={xi}  z={z:.6f}  prob={p:.6f}  pred={pred}  actual={yi}")

    return w, b

def predict(X , y , preds,lr=0.5, epochs=100, threshold=0.5, verbose=True, print_every=1):
    w , b = logistic_regression_gd(X, y, lr=lr, epochs=epochs, verbose=verbose, print_every=print_every , threshold=threshold)
    print(w, "\n" , b)
    z = []
    for xi in preds:
        zi = 0
        for wj, xij in zip(w, xi):
            zi += wj * xij
        z.append(zi + b)
    print("\nLogits (z):", z)
    p = [1 / (1 + math.exp(-zi)) for zi in z]
    pred = [1 if pi >= threshold else 0 for pi in p]
    print("\nPredictions:" , pred)
    return pred

def Scikit_predict(x, y, preds , lr=0.5, epochs=100, threshold=0.5, verbose=True):
    model = LogisticRegression(verbose=verbose, max_iter=epochs, C=1/lr)
    model.fit(x, y)
    pred = model.predict(preds)
    print("\nScikit-learn Predictions:", pred)
    return pred

if __name__ == "__main__":
    X2 = [[1, 1], [2, 1], [2, 2], [3, 2]]
    y2 = [0, 0, 1, 1]
    predict(X2, y2, lr=0.9, epochs=15, verbose=True, print_every=1, preds=[[1, 1], [2, 1], [2, 2], [3, 2], [4, 3]])
    print("\n" + "=" * 70)
    print("Using Scikit-learn for comparison:")
    Scikit_predict(X2, y2, [[1, 1], [2, 1], [2, 2], [3, 2], [4, 3]] , lr=0.9, epochs=15, verbose=True)