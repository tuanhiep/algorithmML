#  Copyright (c) 2020. Tuan-Hiep TRAN
from sklearn.tree import DecisionTreeRegressor


class MyDecisionTreeRegressor:
    def __init__(self, max_depth):
        self.model = DecisionTreeRegressor(max_depth=max_depth)
        self.name="Decision Tree Regressor"

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)
