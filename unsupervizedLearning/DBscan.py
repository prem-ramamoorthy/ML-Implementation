import math
from sklearn.cluster import DBSCAN


def modelprediction(values , eps=1.5, min_samples=2 ):
    model = DBSCAN(eps=eps , min_samples=min_samples)
    model.fit(values) 
    return model.labels_

def calculated(values , eps=1.5, min_samples=2) :
    listOfdistances = [] 
    nfeatures = len(values[0])
    for i in range(len(values)):
        l = []
        for j in range(i , len(values)) :
                dist = 0.0
                for k in range(nfeatures) :
                    dist += math.pow((values[i][k] - values[j][k]), 2)
                l.append(math.sqrt(dist))
        listOfdistances.append(l)

    horizontal_distance = []
    for d in range(len(listOfdistances)):
        diag = []
        for i in range(d + 1):
            j = d - i
            diag.append(listOfdistances[i][j])
        horizontal_distance.append(diag)

    clusters = {}
    n = len(listOfdistances)
    for i in range(n):
        l = []
        for j in range(len(listOfdistances[i])) :
            if j == 0 : 
                continue
            elif listOfdistances[i][j]<=eps :
                l.append( j+1+i )
        clusters.update({str(i+1) : l})
    for i in range(n):
        l = []
        for j in range(len(horizontal_distance[i])) :
            if j == len(horizontal_distance[i]) -1 : 
                continue
            elif horizontal_distance[i][j]<=eps :
                l.append(n-(n-j-1)) 
        for k in l :
            clusters[str(i+1)].append(k)
    core_points = [] 
    cluster_points = [-1 for i in range(len(values))]
    
    for i,v in clusters.items() :
        if(len(v)+1)>= min_samples :
            core_points.append(int(i))
    ind = 0
    for i, v in clusters.items() :
        if int(i) in core_points :
            cluster_points[ind] = int(i)
        else :
            for core in core_points :
                if core in v :
                    cluster_points[ind] = core
        ind+=1
    return cluster_points


        

print(calculated([[3,7] ,[4,6] ,[5,5] ,[6,4] ,[7,3]  ,[6,2] ,[7,2]  ,[8,4] ,[3,3]  ,[2,6] ,[3,5]  ,[2,4]] , 1.9 , 4) )
print(modelprediction([[3,7] ,[4,6] ,[5,5] ,[6,4] ,[7,3]  ,[6,2] ,[7,2]  ,[8,4] ,[3,3]  ,[2,6] ,[3,5]  ,[2,4]] , 1.9 , 4)) 