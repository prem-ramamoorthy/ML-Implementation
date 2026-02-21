import numpy as np

X = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
])

U, S, Vt = np.linalg.svd(X, full_matrices=False)

print("U shape:", U.shape)     # (n_samples, r) used for analysis of directions of samples
print("S shape:", S.shape)     # (r,) used for analysis of importance of each direction
print("Vt shape:", Vt.shape)   # (r, n_features) used for analysis of directions of features

print("U:\n", U)
print("S:\n", S)
print("Vt:\n", Vt)