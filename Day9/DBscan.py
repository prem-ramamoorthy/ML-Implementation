from sklearn.cluster import DBSCAN
import math

def sklearn_dbscan_clustering(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)
    return dbscan.labels_


def dist(a, b):
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s)

def region_query(points, i, eps):
    nbrs = []
    for j in range(len(points)):
        if dist(points[i], points[j]) <= eps:
            nbrs.append(j)
    return nbrs

def dbscan(points, eps, min_pts):
    n = len(points)
    labels = [None] * n
    cluster_id = 0

    for i in range(n):
        if labels[i] is not None:
            continue

        neighbors = region_query(points, i, eps)

        if len(neighbors) < min_pts:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        queue = neighbors[:]

        while queue:
            j = queue.pop(0)

            if labels[j] == -1:
                labels[j] = cluster_id

            if labels[j] is not None:
                continue

            labels[j] = cluster_id
            j_neighbors = region_query(points, j, eps)

            if len(j_neighbors) >= min_pts:
                for t in j_neighbors:
                    if t not in queue:
                        queue.append(t)

        cluster_id += 1

    return labels

data = [
    [1.0, 1.0], [1.2, 1.1], [0.9, 1.0],       # cluster 0
    [8.0, 8.0], [8.2, 8.1], [7.9, 8.0],       # cluster 1
    [4.0, 4.0]                                 # noise
]

labels = dbscan(data, eps=0.35, min_pts=3)

for p, l in zip(data, labels):
    print(p, "->", l)


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_moons
    import matplotlib.pyplot as plt

    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    labels = sklearn_dbscan_clustering(X, eps=0.2, min_samples=5)
    print("DBSCAN Labels:", labels)
    print("Number of clusters:", len(set(labels)) - (1 if -1 in labels else 0))
    print("Number of noise points:", list(labels).count(-1))
    print("Unique labels:" , set(labels))
    print("Cluster sizes:", {label: list(labels).count(label) for label in set(labels)})

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title("DBSCAN Clustering")
    plt.show()