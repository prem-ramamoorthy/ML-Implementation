from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt

def sklearn_hierarchical_clustering(data, n_clusters):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical.fit(data)
    return hierarchical.labels_

def dendrogram_clustering(labels):
    Z = linkage(labels, method='ward')
    dendrogram(Z)
    plt.show()
    return fcluster(Z, t=4, criterion='maxclust')

def euclidean(a, b) :
    s = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        s += diff * diff
    return sqrt(s)


def centroid(points ) :
    d = len(points[0])
    c = [0.0] * d
    for p in points:
        for k in range(d):
            c[k] += p[k]
    n = float(len(points))
    for k in range(d):
        c[k] /= n
    return c


def cluster_distance(
    A_idx ,
    B_idx ,
    X ,
    linkage ,
    point_dist ,
) :
    A = [X[i] for i in A_idx]
    B = [X[i] for i in B_idx]

    if linkage == "single":
        best = float("inf")
        for a in A:
            for b in B:
                d = point_dist(a, b)
                if d < best:
                    best = d
        return best

    if linkage == "complete":
        best = 0.0
        for a in A:
            for b in B:
                d = point_dist(a, b)
                if d > best:
                    best = d
        return best

    if linkage == "average":
        total = 0.0
        cnt = 0
        for a in A:
            for b in B:
                total += point_dist(a, b)
                cnt += 1
        return total / cnt

    if linkage == "centroid":
        ca = centroid(A)
        cb = centroid(B)
        return point_dist(ca, cb)

    if linkage == "ward":
        ca = centroid(A)
        cb = centroid(B)
        sq = 0.0
        for k in range(len(ca)):
            diff = ca[k] - cb[k]
            sq += diff * diff
        na = float(len(A))
        nb = float(len(B))
        factor = (na * nb) / (na + nb)
        return sqrt(factor * sq)

    raise ValueError(f"Unknown linkage: {linkage}")


def agglomerative(
    X ,
    n_clusters ,
    linkage ,
    point_dist = "euclidean",
    return_history = True,
) :
    n = len(X)
    if n == 0:
        raise ValueError("X is empty")
    if n_clusters < 1 or n_clusters > n:
        raise ValueError("n_clusters must be in [1, n]")
    linkage = linkage.lower()
    
    if isinstance(point_dist, str):
        if point_dist == "euclidean":
            point_dist = euclidean
        else:
            raise ValueError(f"Unknown point_dist: {point_dist}")
 
    clusters  = {i: [i] for i in range(n)}
    active_ids = list(clusters.keys())
 
    dist_cache = {}

    def get_dist(id1 , id2 ) :
        a, b = (id1, id2) if id1 < id2 else (id2, id1)
        key = (a, b)
        if key in dist_cache:
            return dist_cache[key]
        d = cluster_distance(clusters[a], clusters[b], X, linkage, point_dist)
        dist_cache[key] = d
        return d

    history  = []
    next_cluster_id = n

    while len(active_ids) > n_clusters:
        best_pair = None
        best_dist = float("inf")

        L = len(active_ids)
        for i in range(L):
            for j in range(i + 1, L):
                idA = active_ids[i]
                idB = active_ids[j]
                d = get_dist(idA, idB)
                if d < best_dist:
                    best_dist = d
                    best_pair = (idA, idB)

        idA, idB = best_pair
 
        new_id = next_cluster_id
        next_cluster_id += 1

        clusters[new_id] = clusters[idA] + clusters[idB]
 
        active_ids = [cid for cid in active_ids if cid not in (idA, idB)]
        active_ids.append(new_id)
 
        if return_history:
            history.append(
                {
                    "merge": (idA, idB),
                    "dist": best_dist,
                    "new_id": new_id,
                    "size": len(clusters[new_id]),
                }
            )
 
    labels = [-1] * n
    active_ids_sorted = sorted(active_ids)
    for label, cid in enumerate(active_ids_sorted):
        for idx in clusters[cid]:
            labels[idx] = label

    return labels, (history if return_history else None)

X = [
    [1.0, 1.0],
    [1.2, 0.9],
    [0.8, 1.1],
    [5.0, 5.0],
    [5.2, 4.9],
    [4.8, 5.1],
    [9.0, 1.0],
    [9.2, 1.1],
    [8.8, 0.9],
]

for link in ["single", "complete", "average", "centroid", "ward"]:
    labels, hist = agglomerative(X, n_clusters=3, linkage=link, return_history=True)
    print(f"\nLinkage: {link}")
    print("Labels:", labels)
    print("First 3 merges:", hist[:3])


if __name__ == "__main__":
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    labels = sklearn_hierarchical_clustering(data, n_clusters=4)
    print(labels)
    dendrogram_labels = dendrogram_clustering(labels)
    print(dendrogram_labels)