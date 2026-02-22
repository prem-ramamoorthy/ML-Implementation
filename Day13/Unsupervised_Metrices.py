from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from matplotlib import pyplot as plt

X, labels_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

def evaluate_clustering(X, labels_true=None):
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels_pred = kmeans.fit_predict(X)
    print("Predicted labels:", labels_pred)
    return labels_pred

xs = []

def nks(X, labels_true=None):
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels_pred = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels_pred)
        dbi = davies_bouldin_score(X, labels_pred)
        ch  = calinski_harabasz_score(X, labels_pred)
        xs.append((k, sil, dbi, ch))
        print(f"k={k}: Silhouette Score={sil:.3f}, Davies-Bouldin Index={dbi:.3f}, Calinski-Harabasz Index={ch:.3f}")
nks(X, labels_true)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot([x[0] for x in xs], [x[1] for x in xs], marker='o')
plt.title('Silhouette Score vs k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')

plt.subplot(1, 3, 2)
plt.plot([x[0] for x in xs], [x[2] for x in xs], marker='o')
plt.title('Davies-Bouldin Index vs k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index')

plt.subplot(1, 3, 3)
plt.plot([x[0] for x in xs], [x[3] for x in xs], marker='o')
plt.title('Calinski-Harabasz Index vs k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Calinski-Harabasz Index')
plt.tight_layout()
plt.show()

labels_pred = evaluate_clustering(X)

sil = silhouette_score(X, labels_pred) # si = silhouette index si = ai - bi / max(ai, bi) ai = average distance to points in the same cluster bi = average distance to points in the nearest cluster cluster, lower is better
dbi = davies_bouldin_score(X, labels_pred) # dbi = davies-bouldin index dbi = 1/n * sum(max((si + sj) / dij)) si = average distance to points in the same cluster sj = average distance to points in the nearest cluster dij = distance between cluster centers
# used to evaluate the quality of clustering results, lower is better
ch  = calinski_harabasz_score(X, labels_pred) # ch = calinski-harabasz index ch = (tr / (k - 1)) / (tw / (n - k)) tr = total between-cluster variance tw = total within-cluster variance k = number of clusters n = number of samples
# used to evaluate the quality of clustering results, higher is better
print(f"Silhouette Score: {sil:.3f}")
print(f"Davies-Bouldin Index: {dbi:.3f}")
print(f"Calinski-Harabasz Index: {ch:.3f}")

# if true labels exist
ari = adjusted_rand_score(labels_true, labels_pred) # ari = adjusted rand index ari = (RI - Expected RI) / (max(RI) - Expected RI) RI = (TP + TN) / (TP + FP + FN + TN) TP = true positives FP = false positives FN = false negatives TN = true negatives
nmi = normalized_mutual_info_score(labels_true, labels_pred) # nmi = normalized mutual information nmi = I(X; Y) / sqrt(H(X) * H(Y)) I(X; Y) = mutual information H(X) = entropy of X H(Y) = entropy of Y

print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Normalized Mutual Information: {nmi:.3f}")

print("Optimal number of clusters (k) can be determined by analyzing the plots of the metrics. Look for the 'elbow' point in the Silhouette Score and Calinski-Harabasz Index plots, and the lowest point in the Davies-Bouldin Index plot.")
optimal_k_silhouette = max(xs, key=lambda x: x[1])[0]
optimal_k_dbi = min(xs, key=lambda x: x[2])[0]
optimal_k_ch = max(xs, key=lambda x: x[3])[0]
print(f"Optimal k based on Silhouette Score: {optimal_k_silhouette}")
print(f"Optimal k based on Davies-Bouldin Index: {optimal_k_dbi}")
print(f"Optimal k based on Calinski-Harabasz Index: {optimal_k_ch}")
voted = max([optimal_k_silhouette, optimal_k_dbi, optimal_k_ch])
print(f"Voted optimal k: {voted}")
print(f"In this case, k={voted} seems to be the optimal number of clusters based on the metrics and the plots.")
