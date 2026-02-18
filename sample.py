import numpy as np
from sklearn.feature_selection import chi2

# X must be non-negative
X = np.array([
    [1, 0, 3],
    [2, 1, 0],
    [0, 1, 2],
    [3, 0, 1]
])

y = np.array([0, 1, 0, 1])   # target classes

chi_scores, p_values = chi2(X, y)

print("Chi-square scores:", chi_scores)
print("P-values:", p_values)
