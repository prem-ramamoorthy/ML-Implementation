from sklearn.neighbors import KNeighborsClassifier
import math 

def modelprediction(x,y,pred , n = 5):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x,y)
    return knn.predict(pred)

def get_ranks(lst, reverse=False):
    sorted_list = sorted(enumerate(lst), key=lambda x: x[1], reverse=reverse)
    ranks = [0] * len(lst)
    for rank, (original_index, _) in enumerate(sorted_list):
        ranks[original_index] = rank + 1  # +1 for 1-based rank (optional)
    return ranks

def Knn(x,y,pred , n = 5):
    nfeatures = len(x[0])
    distance = []
    for i in range(len(x)) : 
        sum = 0
        for j in range(nfeatures) :
            sum+= math.pow((pred[0][j] - x[i][j]) , 2 )
        distance.append(math.sqrt(sum))
    rank = get_ranks(distance)
    pred_values = [ i+1 for i in range(n)]
    predictions = []
    for i in range(len(pred_values)) :
        predictions.append(y[rank.index(pred_values[i])][0])
        pred_values[i] = distance[rank.index(pred_values[i])]
    s = set(predictions)
    s = {predictions.count(i): i for i in s}
    return s[max(s.keys())]

x = [[5.3, 3.7], [5.1, 3.8], [7.2, 3.0], [5.4, 3.4], [5.1, 3.3],
     [5.4, 3.9], [7.4, 2.8], [6.1, 2.8], [7.3, 2.9], [6.0, 2.7],
     [5.8, 2.8], [6.3, 2.3], [5.1, 2.5], [6.3, 2.5], [5.5, 2.4]]

y = [["s"], ["s"], ["v"], ["s"], ["s"],
     ["s"], ["v"], ["ve"], ["v"], ["ve"],
     ["v"], ["ve"], ["ve"], ["ve"], ["ve"]]

print(Knn(x,y,[[5.2 , 3.1]] , 5))
# print(modelprediction(x,y,[[5.2 , 3.1]] , 5))

# [0.608276253029822, 0.7071067811865474, 2.0024984394500787, 0.3605551275463989, 0.22360679774997896, 0.8246211251235319, 2.220360331117452, 0.9486832980505134, 2.109502310972898, 0.8944271909999156, 0.6708203932499367, 1.3601470508735443, 0.6082762530298221, 1.2529964086141665, 0.761577310586391]
# [3, 6, 13, 2, 1, 8, 15, 10, 14, 9, 5, 12, 4, 11, 7]
# [0.22360679774997896, 0.3605551275463989, 0.608276253029822, 0.6082762530298221, 0.6708203932499367]