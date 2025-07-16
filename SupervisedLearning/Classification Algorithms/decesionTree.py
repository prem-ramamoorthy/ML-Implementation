import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    data = pd.DataFrame(pd.read_csv("SampleData.csv"))
    columns = data.columns

    attributes = []
    result = []
    le = LabelEncoder()

    for i in columns:
        if (i == "AccountNumber" or i == "OPENING_DATE" or i == "Tenure" or i == "TXN_AMOUNT" or i == "INACTIVE"):
            continue
        elif (i == "churn"):
            result.append(i)
        else:
            attributes.append(i)

    for i in attributes:
        if data[i].dtype == 'object':
            data[i] = le.fit_transform(data[i])

    target_column_name = result[0]
    if data[target_column_name].dtype == 'object':
        data[target_column_name] = le.fit_transform(data[target_column_name])
    
    churn_classes = le.classes_ 


    x = data[attributes]
    y = data[result]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(x_train, y_train)

    y_pred_clf = clf.predict(x_test)
    print("Predicted values (first 10):", y_pred_clf[:10].flatten())

    print("\n--- Model Evaluation ---")

    accuracy = accuracy_score(y_test, y_pred_clf)
    print(f"\nAccuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred_clf)
    print("\nConfusion Matrix:")
    print(cm)


    optimized_feature_names = [name.replace('_', ' ').title() for name in attributes]
    plt.figure(figsize=(20, 15)) 
    plot_tree(
        clf, 
        feature_names=optimized_feature_names,
        class_names=churn_classes.astype(str),
        filled=False,
        rounded=True,
        proportion=True,
        fontsize=10 , 
        max_depth = 5 
    )
    plt.title("Decision Tree for Customer Churn Prediction")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                xticklabels=churn_classes, yticklabels=churn_classes) 
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Churn Prediction')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_clf, target_names=churn_classes)) 

    precision = precision_score(y_test, y_pred_clf, average='weighted')
    recall = recall_score(y_test, y_pred_clf, average='weighted')
    f1 = f1_score(y_test, y_pred_clf, average='weighted')

    print(f"\nIndividual Metrics (Weighted Average):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")