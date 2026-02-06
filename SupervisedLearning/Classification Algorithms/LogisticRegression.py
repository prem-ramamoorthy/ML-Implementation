# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# import math

# def LogisticRegressionModel(x,y,pred):
#     pass

# def modelPrediction(x,y,xpred):
#     model = LogisticRegression()
#     x = [ [i] for i in x]
#     model.fit(x , y)
#     return model.predict([[xpred]])[0]

# x = [29, 15 , 33 , 28 , 39 ]
# y = [0, 0, 1, 1, 1]
# xpred = 33
# print("Model Prediction Value :" , modelPrediction(x,y,xpred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

dataframe = pd.read_csv("SampleData.csv") 
dataframe = dataframe.dropna()

columns_to_drop_early = ["OPENING_DATE", "AccountNumber", "Tenure", "current balance", "TXN_AMOUNT"]
dataframe = dataframe.drop(columns=columns_to_drop_early, errors='ignore')

y_raw = dataframe['churn']
X_raw = dataframe.drop(columns=['churn'])
categorical_features_for_X = ["type of account", "INACTIVE", "Mobile user", "has ATM CARD", "CURRENCY", "Job type", "GENDER"]

x_encoded_features = pd.get_dummies(X_raw, columns=categorical_features_for_X, drop_first=True, dtype='int')
label_encoder = LabelEncoder()
y_encoded_target = label_encoder.fit_transform(y_raw)
x = x_encoded_features
y = y_encoded_target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = clf.score(x_test, y_test)
print("accuracy is ")
print(accuracy)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_) 
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.title('confusion matrix for logistic regression')
plt.show()