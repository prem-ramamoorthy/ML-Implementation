import math
from collections import Counter, defaultdict

def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

def information_gain(rows, feature, target_key):
    base = entropy([r[target_key] for r in rows])

    groups = defaultdict(list)
    for r in rows:
        groups[r[feature]].append(r)

    cond = 0.0
    n = len(rows)
    for grp in groups.values():
        cond += (len(grp) / n) * entropy([r[target_key] for r in grp])

    return base - cond

def majority_class(rows, target_key):
    labels = [r[target_key] for r in rows]
    return Counter(labels).most_common(1)[0][0]

def id3(rows, features, target_key):
    labels = [r[target_key] for r in rows]
    if len(set(labels)) == 1:
        return labels[0]
    if not features:
        return majority_class(rows, target_key)

    gains = [(information_gain(rows, f, target_key), f) for f in features]
    gains.sort(reverse=True)
    best_gain, best_feature = gains[0]

    if best_gain <= 1e-12:
        return majority_class(rows, target_key)

    tree = {"feature": best_feature, "branches": {}}
    groups = defaultdict(list)
    for r in rows:
        groups[r[best_feature]].append(r)

    remaining = [f for f in features if f != best_feature]
    for val, grp in groups.items():
        tree["branches"][val] = id3(grp, remaining, target_key)

    return tree

def predict_id3(tree, x, default=None):
    if not isinstance(tree, dict):
        return tree
    f = tree["feature"]
    v = x.get(f)
    branches = tree["branches"]
    if v in branches:
        return predict_id3(branches[v], x, default)
    return default  

def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "->", tree)
        return
    print(indent + "Feature:", tree["feature"])
    for val, subtree in tree["branches"].items():
        print(indent + f"Value: {val}")
        print_tree(subtree, indent + "  ")
        
def scikit_predict(data , algo = "id3"):
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    X = np.array([[0 if r["Outlook"] == "Sunny" else 1 if r["Outlook"] == "Overcast" else 2,
                   0 if r["Humidity"] == "Normal" else 1] for r in data])
    y = np.array([0 if r["Play"] == "No" else 1 for r in data])
    if algo == "id3" or algo == "c4.5":
        critirion = 'entropy'
    elif (algo == "cart"):
        critirion = 'gini'
    else:
        raise ValueError("Unsupported algorithm")
    clf = DecisionTreeClassifier(criterion=critirion)
    clf.fit(X, y)
    return clf.predict([[0, 0]])[0]

data = [
    {"Outlook": "Sunny",    "Humidity": "Normal",   "Play": "No"},
    {"Outlook": "Sunny",    "Humidity": "High",   "Play": "Yes"},
    {"Outlook": "Overcast", "Humidity": "High",   "Play": "Yes"},
    {"Outlook": "Rainy",    "Humidity": "Normal", "Play": "Yes"},
    {"Outlook": "Rainy",    "Humidity": "High", "Play": "No"},
]

features = ["Outlook", "Humidity"]
tree = id3(data, features, target_key="Play")
print_tree(tree)

test = {"Outlook": "Sunny", "Humidity": "Normal"}
print("Prediction:", predict_id3(tree, test, default="Yes"))

print("Scikit-learn Prediction:", "Yes" if scikit_predict(data , algo="id3") == 1 else "No")