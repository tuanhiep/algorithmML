#  Copyright (c) 2020. Tuan-Hiep TRAN
from sklearn.linear_model import RANSACRegressor, LinearRegression


class MyRANSACRegressor:

    def __init__(self, min_sample):
        self.model = RANSACRegressor(LinearRegression(), max_trials=100,
                                     min_samples=min_sample, loss='absolute_loss',
                                     residual_threshold=5.0, random_state=1)
        self.name = "RANSAC"

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)
