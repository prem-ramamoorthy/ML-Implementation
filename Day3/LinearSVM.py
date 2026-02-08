import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):

                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0

                else:
                    dw = 2 * self.lambda_param * self.w - y_[idx] * x_i
                    db = -y_[idx]

                self.w = self.w - self.lr * dw
                self.b = self.b - self.lr * db

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output) # The np.sign function returns -1 for negative values, 0 for zero, and 1 for positive values. In this case, it will return -1 for samples predicted as class -1 and 1 for samples predicted as class 1.
    
def scikitlearn_svm(X, y , c):
    from sklearn import svm
    clf = svm.SVC(kernel='linear' , C=c) # kernel='linear' specifies that we want to use a linear kernel, and C=c is the regularization parameter that controls the trade-off between achieving a low error on the training data and minimizing the model complexity for better generalization to unseen data.
    clf.fit(X, y)
    return clf

if __name__ == "__main__":
    X = np.array([
        [3, 3],
        [4, 3],
        [3, 4],
        [1, 1],
        [2, 1],
        [1, 2]
    ])

    y = np.array([1, 1, 1, -1, -1, -1])

    model = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000) # learning_rate is the step size for updating weights, lambda_param is the regularization parameter, and n_iters is the number of iterations for training the model.
    model.fit(X, y)

    predictions = model.predict(X)
    print("Predictions:", predictions)
    
    # Using scikit-learn's SVM for comparison
    clf = scikitlearn_svm(X, y, c=1.0)
    sklearn_predictions = clf.predict(X)
    print("Scikit-learn Predictions:", sklearn_predictions)