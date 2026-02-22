from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt

X_train, y_train = make_classification(n_samples=300, n_features=20, n_classes=2, random_state=42)
X_test, y_test = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=24)

def evaluate_classification(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Predicted labels:", y_pred)
    return y_pred

y_pred = evaluate_classification(X_train, y_train, X_test, y_test)

acc = accuracy_score(y_test, y_pred) # acc = accuracy score acc = (TP + TN) / (TP + TN + FP + FN) TP = true positives TN = true negatives FP = false positives FN = false negatives
prec = precision_score(y_test, y_pred) # prec = precision score prec = TP / (TP + FP) TP = true positives FP = false positives
rec = recall_score(y_test, y_pred) # rec = recall score rec = TP / (TP + FN) TP = true positives FN = false negatives
f1 = f1_score(y_test, y_pred) # f1 = f1 score f1 = 2 * (prec * rec) / (prec + rec) prec = precision score rec = recall score
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1 Score: {f1:.3f}")
cm = confusion_matrix(y_test, y_pred) # cm = confusion matrix cm = [[TP, FP], [FN, TN]] TP = true positives FP = false positives FN = false negatives TN = true negatives
print("Confusion Matrix:")
print(cm)
report = classification_report(y_test, y_pred) # report = classification report report = precision, recall, f1-score for each class
print("Classification Report:")
print(report)

plt.figure(figsize=(6, 4))
plt.bar(['Accuracy', 'Precision', 'Recall', 'F1 Score'], [acc, prec, rec, f1], color=['blue', 'orange', 'green', 'red'])
plt.ylim(0, 1)
plt.title('Classification Metrics')
plt.ylabel('Score')
plt.show()

print("\n" + "="*50 + "\n")

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

X_train, y_train = make_classification(n_samples=300, n_features=20, n_classes=1, random_state=42)
X_test, y_test = make_classification(n_samples=100, n_features=20, n_classes=1, random_state=24)

def evaluate_regression(X_train, y_train, X_test, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("Predicted values:", y_pred)
    return y_pred
y_pred = evaluate_regression(X_train, y_train, X_test, y_test)

mse = mean_squared_error(y_test, y_pred) # mse = mean squared error mse = (1/n) * sum((y_true - y_pred)^2) n = number of samples y_true = true values y_pred = predicted values
mae = mean_absolute_error(y_test, y_pred) # mae = mean absolute error mae = (1/n) * sum(|y_true - y_pred|) n = number of samples y_true = true values y_pred = predicted values
r2 = r2_score(y_test, y_pred) # r2 = r2 score r2 = 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)) y_true = true values y_pred = predicted values mean(y_true) = mean of true values

print(f"Mean Squared Error: {mse:.3f}")
print(f"Mean Absolute Error: {mae:.3f}")
print(f"R2 Score: {r2:.3f}")