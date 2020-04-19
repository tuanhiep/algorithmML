#  Copyright (c) 2020. Tuan-Hiep TRAN

import numpy as np


class MyLinearRegressionNE:
    def __init__(self):
        self.w = None
        self.name = "Linear Regression Normal Equation"

    def fit(self, p_X, p_Y):
        X = np.zeros((p_X.shape[0], p_X.shape[1] + 1))
        X[:, :-1] = p_X
        # X = p_X in case without intercept term
        Y = p_Y.reshape(p_Y.shape[0], 1)
        self.w = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(Y))

    def predict(self, p_X):
        X = np.zeros((p_X.shape[0], p_X.shape[1] + 1))
        X[:, :-1] = p_X
        # X = p_X in case without intercept term
        y_predict = X.dot(self.w)
        return y_predict
