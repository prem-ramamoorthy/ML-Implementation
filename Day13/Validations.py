from sklearn.model_selection import cross_val_score , StratifiedKFold, LeaveOneOut, GroupKFold, GroupShuffleSplit, TimeSeriesSplit
from sklearn.datasets import make_classification

from sklearn.ensemble import AdaBoostClassifier

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
model = AdaBoostClassifier(n_estimators=100, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print("StratifiedKFold Cross-Validation Accuracy Scores:", scores)
print("Mean Cross-Validation Accuracy:", scores.mean())

# loo = LeaveOneOut()
# loo_scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
# print("Leave-One-Out Cross-Validation Accuracy Scores:", loo_scores)
# print("Mean Leave-One-Out Cross-Validation Accuracy:", loo_scores.mean())

groups = [i // 100 for i in range(1000)]
group_kfold = GroupKFold(n_splits=5)
group_scores = cross_val_score(model, X, y, cv=group_kfold.split(X, y, groups), scoring='accuracy')
print("Group K-Fold Cross-Validation Accuracy Scores:", group_scores)
print("Mean Group K-Fold Cross-Validation Accuracy:", group_scores.mean())

group_shuffle = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
group_shuffle_scores = cross_val_score(model, X, y, cv=group_shuffle.split(X, y, groups), scoring='accuracy')
print("Group Shuffle Split Cross-Validation Accuracy Scores:", group_shuffle_scores)
print("Mean Group Shuffle Split Cross-Validation Accuracy:", group_shuffle_scores.mean())

time_series_split = TimeSeriesSplit(n_splits=5)
time_series_scores = cross_val_score(model, X, y, cv=time_series_split, scoring='accuracy')
print("Time Series Split Cross-Validation Accuracy Scores:", time_series_scores)
print("Mean Time Series Split Cross-Validation Accuracy:", time_series_scores.mean())