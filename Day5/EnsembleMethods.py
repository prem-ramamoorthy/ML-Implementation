from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier , StackingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

baggingBase = DecisionTreeClassifier(random_state=42)
baggingModel = BaggingClassifier(
    estimator=baggingBase,
    n_estimators=200,
    max_samples=0.8,
    max_features=1.0,
    bootstrap=True,
    random_state=42
)

baggingModel.fit(X_train, y_train)
baggingPred = baggingModel.predict(X_test)
print("Bagging Accuracy:", accuracy_score(y_test, baggingPred))

rfModel = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

rfModel.fit(X_train, y_train)
pred_rf = rfModel.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, pred_rf))

AdaBoostbase = DecisionTreeClassifier(max_depth=1, random_state=42)
AdaBoostmodel = AdaBoostClassifier(
    estimator=AdaBoostbase,
    n_estimators=300,
    learning_rate=0.5,
    random_state=42
)
AdaBoostmodel.fit(X_train, y_train)
pred_ada = AdaBoostmodel.predict(X_test)
print("AdaBoost Accuracy:", accuracy_score(y_test, pred_ada))

estimators = [
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ("svm", SVC(probability=True, random_state=42)),
]

final = LogisticRegression(max_iter=5000)

stackingModel = StackingClassifier(
    estimators=estimators,
    final_estimator=final,
    passthrough=False # If True, the original features are concatenated with the predictions of the base estimators and passed to the final estimator. If False, only the predictions of the base estimators are passed to the final estimator.
)

stackingModel.fit(X_train, y_train)
stackingPred = stackingModel.predict(X_test)
print("Stacking Accuracy:", accuracy_score(y_test, stackingPred))

Votingmodel = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=5000)),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ("svm", SVC(probability=True, random_state=42)),
    ],
    voting="soft" # soft - probability based voting, hard - majority voting
)

Votingmodel.fit(X_train, y_train)
votingPred = Votingmodel.predict(X_test)
print("Voting Accuracy:", accuracy_score(y_test, votingPred))