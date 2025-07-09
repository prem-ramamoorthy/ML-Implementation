import pandas as  pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
        
    data = pd.DataFrame(pd.read_csv("SampleData.csv"))

    columns = data.columns

    attributes = []
    result = []

    le = LabelEncoder()
    
    for i in columns :
        if (i == "AccountNumber"):
            continue
        elif(i == "OPENING_DATE"):
            continue
        elif(i == "Tenure"):
            continue
        elif(i == "TXN_AMOUNT"):
            continue
        elif(i=="churn"):
            result.append(i)
        else:
            attributes.append(i)
    
    for i in attributes :
        data[i] = le.fit_transform(data[i])

    for i in result :
        data[i] = le.fit_transform(data[i])

    x = data[attributes]
    y = data[result]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    clf = DecisionTreeClassifier(random_state=42) 
    clf.fit(x_train, y_train)  
    y_pred_clf = clf.predict(x_test) 
    print(y_pred_clf)
