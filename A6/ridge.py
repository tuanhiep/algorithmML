#  Copyright (c) 2020. Tuan-Hiep TRAN
from sklearn.linear_model import Ridge


class MyRidge:
    def __init__(self, alpha):
        self.ridge = Ridge(alpha=alpha)
        self.name="RIDGE"

    def fit(self, X, Y):
        self.ridge.fit(X, Y)

    def predict(self, X):
        return self.ridge.predict(X)
