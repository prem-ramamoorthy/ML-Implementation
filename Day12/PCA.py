import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

X, y_true = make_blobs(
    n_samples=600,
    n_features=5,
    centers=3,
    cluster_std=[1.0, 2.5, 0.8],
    random_state=42
)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Original shape:", X.shape)
print("Transformed shape:", X_pca.shape)
print("Original data (first 5 samples):\n", X[:5])
print("PCA-transformed data (first 5 samples):\n", X_pca[:5])
print("Principal components:\n", pca.components_)
print("Explained variance ratio:", pca.explained_variance_ratio_)
plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, s=20)
plt.title("PCA Projection of Blobs Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()