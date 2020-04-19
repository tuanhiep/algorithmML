#  Copyright (c) 2020. Tuan-Hiep TRAN
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class MyLinearRegression:
    def __init__(self):
        self.model = LinearRegression()
        self.name="Linear Regression"

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    def linear_regression_plot(self, X, y):
        plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
        plt.plot(X, self.model.predict(X), color='black', lw=2)


