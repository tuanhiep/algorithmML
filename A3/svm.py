#  Copyright (c) 2020. Tuan Hiep TRAN
from sklearn.svm import SVC


class MySVM:
    def __init__(self, kernel):
        self.svm = SVC(kernel=kernel)

    def fit(self, X, y):
        self.svm.fit(X, y)

    def predict(self, X):
        return self.svm.predict(X)

    def score(self, X, y):
        return self.svm.score(X, y)
