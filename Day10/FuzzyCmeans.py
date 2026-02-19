import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

X = np.array([
    [1.0, 1.0], [1.2, 0.9], [0.8, 1.1],     # cluster-ish A
    [5.0, 5.0], [5.2, 4.8], [4.9, 5.1],     # cluster-ish B
    [9.0, 1.0], [8.7, 1.3], [9.2, 0.8],     # cluster-ish C
], dtype=float)

data = X.T  # (2, N)

n_clusters = 3
m = 2.0             # fuzziness (usually 1.5 to 2.5)
error = 1e-5
maxiter = 1000

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data=data,
    c=n_clusters,
    m=m,
    error=error,
    maxiter=maxiter,
    init=None,
    seed=42
)

labels = np.argmax(u, axis=0)

print("Centers:\n", cntr)
print("FPC (higher is better):", fpc)

new_points = np.array([
    [1.1, 1.0],
    [5.1, 5.1],
    [9.0, 1.2],
], dtype=float)

result = fuzz.cluster.cmeans_predict(
    test_data=new_points.T,
    cntr_trained=cntr,
    m=m,
    error=error,
    maxiter=maxiter
)
u_new, _, _, _, _, _ = result
print("\nNew points membership (rows=clusters, cols=points):\n", u_new)
print("New points hard labels:", np.argmax(u_new, axis=0))

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=80)
plt.scatter(cntr[:, 0], cntr[:, 1], marker="X", s=250)
plt.title("Fuzzy C-Means (colored by max membership)")
plt.show()