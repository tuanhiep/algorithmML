#  Copyright (c) 2020. Tuan-Hiep TRAN
from sklearn.linear_model import Lasso


class MyLasso:
    def __init__(self, alpha):
        self.lasso = Lasso(alpha=alpha)
        self.name="LASSO"

    def fit(self, X, Y):
        self.lasso.fit(X, Y)

    def predict(self, X):
        return self.lasso.predict(X)
