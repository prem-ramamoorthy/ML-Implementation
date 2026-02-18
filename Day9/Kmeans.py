from sklearn.cluster import KMeans
import math
import random

def sklearn_kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_

def euclidean(a, b):
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s)

def mean_point(points, dim):
    if not points:
        return None
    centroid = [0.0] * dim
    for p in points:
        for i in range(dim):
            centroid[i] += p[i]
    n = float(len(points))
    for i in range(dim):
        centroid[i] /= n
    return centroid

def kmeans(points, k, max_iters=100, seed=42):
    if k <= 0:
        raise ValueError("k must be > 0")
    if len(points) < k:
        raise ValueError("Number of points must be >= k")
    dim = len(points[0])
    for p in points:
        if len(p) != dim:
            raise ValueError("All points must have same dimension")

    random.seed(seed)

    indices = list(range(len(points)))
    random.shuffle(indices)
    centroids = [points[i][:] for i in indices[:k]]

    labels = [-1] * len(points)

    for _ in range(max_iters):
        changed = False

        clusters = [[] for _ in range(k)]
        for idx, p in enumerate(points):
            best_j = 0
            best_dist = euclidean(p, centroids[0])

            for j in range(1, k):
                d = euclidean(p, centroids[j])
                if d < best_dist:
                    best_dist = d
                    best_j = j

            if labels[idx] != best_j:
                changed = True
                labels[idx] = best_j

            clusters[best_j].append(p)

        new_centroids = []
        for j in range(k):
            if clusters[j]:
                new_centroids.append(mean_point(clusters[j], dim))
            else:
                new_centroids.append(points[random.randrange(len(points))][:])

        centroids = new_centroids

        if not changed:
            break

    return centroids, labels


data = [
    [1.0, 1.0], [1.5, 2.0], [3.0, 4.0],   # cluster-ish A
    [5.0, 7.0], [3.5, 5.0], [4.5, 5.0], [3.5, 4.5]  # cluster-ish B
]

centroids, labels = kmeans(data, k=2, max_iters=50, seed=7)

print("Final centroids:")
for i, c in enumerate(centroids):
    print(f"  C{i}: {c}")

print("\nPoint -> cluster:")
for p, lbl in zip(data, labels):
    print(f"  {p} -> {lbl}")

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    data = np.random.rand(100, 2)

    n_clusters = 3
    labels = sklearn_kmeans_clustering(data, n_clusters)
    print("Cluster Labels:", labels)

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title('KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()