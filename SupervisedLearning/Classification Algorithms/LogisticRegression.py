import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import math

def LogisticRegressionModel(x,y,pred):
    pass

def modelPrediction(x,y,xpred):
    model = LogisticRegression()
    x = [ [i] for i in x]
    model.fit(x , y)
    return model.predict([[xpred]])[0]

x = [29, 15 , 33 , 28 , 39 ]
y = [0, 0, 1, 1, 1]
xpred = 33
print("Model Prediction Value :" , modelPrediction(x,y,xpred))