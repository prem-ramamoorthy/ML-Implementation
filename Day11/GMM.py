import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, y_true = make_blobs(
    n_samples=600,
    centers=3,
    cluster_std=[1.0, 2.5, 0.8],
    random_state=42
)

gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    random_state=42
)
gmm.fit(X)

labels = gmm.predict(X)

probs = gmm.predict_proba(X)

means = gmm.means_
covs = gmm.covariances_
weights = gmm.weights_

print("Mixing weights:", weights)
print("Means:\n", means)
print("\nSoft probabilities for first 5 points:\n", probs[:5])

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=20)
plt.scatter(means[:, 0], means[:, 1], s=200, marker="X")
plt.title("GMM Clustering (scikit-learn)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
