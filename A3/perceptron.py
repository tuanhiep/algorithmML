#  Copyright (c) 2020. Tuan Hiep TRAN
from sklearn.linear_model import Perceptron

'''
Class define my perceptron classifier 
'''


class MyPerceptron:
    def __init__(self, learning_rate, number_iteration):
        self.perceptron = Perceptron(eta0=learning_rate, max_iter=number_iteration)

    def fit(self, X, y):
        self.perceptron.fit(X, y)

    def predict(self, X):
        return self.perceptron.predict(X)

    def score(self, X, y):
        return self.perceptron.score(X, y)
