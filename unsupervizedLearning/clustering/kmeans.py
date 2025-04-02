import random 
import math
from sklearn.cluster import KMeans
import numpy as np 

def kmeans_computation(values , nclusters = None , centroids = None) :
    if nclusters == None :
         clusters = random.randrange(2,len(values),1)
    if(len(values)< nclusters):
        return "No of cluster should be less than the length of values "
    nfeatures = len(values[0])
    if centroids == None :
        centroids = random.sample(values , nclusters)
    distance = [[] for _ in range(nclusters)]
    clusters = []
    newclusters = []
    while len(newclusters) < 2 or newclusters[-1] != newclusters[-2]:
        distance = [[] for _ in range(nclusters)]
        for cluster in range(nclusters):
            for value in values:
                dist = 0
                for feature in range(nfeatures):
                    dist += math.pow(value[feature] - centroids[cluster][feature], 2)
                distance[cluster].append(math.sqrt(dist))
        v1 = []
        for i in range(len(distance[0])):
            temp = []
            for j in range(len(distance)):
                temp.append(distance[j][i])
            v1.append(temp)
        clusters = [i.index(min(i))+1 for i in v1]
        clustmap = {cluster: [] for cluster in range(1, nclusters + 1)}
        for i in range(len(values)):
            clustmap[clusters[i]].append(values[i])
        l = []
        for _, v in clustmap.items():
            for i in range(len(v[0])):
                temp = []
                for j in range(len(v)) :
                    temp.append(v[j][i])
                l.append(sum(temp)/len(temp))
        newcentroids = []
        for i in range(0, len(l), nfeatures):
            newcentroids.append(l[i:i + nfeatures])
        newclusters.append(clusters)
        centroids = newcentroids
    print("Final centroids of kmeans :" , centroids)
    return newclusters[-1]

def modelPrediction(values , nclusters = None , centroids = None) :
    if nclusters == None :
        clusters = random.randrange(2,len(values),1)
    if(len(values)< nclusters):
        return "No of cluster should be less than the length of values "
    nfeatures = len(values[0])
    if centroids == None :
        centroids = random.sample(values , nclusters)
    centroids = np.array(centroids)
    model = KMeans(n_clusters=nclusters, init=centroids, n_init=10)
    model.fit(values)
    clusters = model.labels_
    centroids = model.cluster_centers_
    print("Final centroids of Model Prediction :" , centroids)
    return [int(cluster+1) for cluster in clusters]

x = [[2,10] , [2,5] , [8,4], [5,8] , [7,5], [6,4] , [1,2] , [4,9]] 
print(kmeans_computation(x,3, [[2,10] ,[5,8] , [1,2]])) 
print(modelPrediction(x,3, [[2,10] ,[5,8] , [1,2]]))