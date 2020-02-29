from A2.sgd import StochasticGD
import numpy as np
import pandas as pd


class OneVsRest:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.sgd_ = []
        self.labels_ = []
        self.accuracy_ = []

    def fit(self, X, y):
        # get distinct labels
        y_df = pd.DataFrame(y)
        self.labels_ = y_df.iloc[:, -1].unique()
        for label in self.labels_:
            target = np.where(y == label, 1, -1)
            sgd = StochasticGD(eta=self.eta, n_iter=self.n_iter)
            sgd.fit(X, target)
            self.sgd_.append((sgd, label))
        return self

    def predict(self, X):
        predicted_ = pd.DataFrame(columns=self.labels_)
        for (sgd, label) in self.sgd_:
            prediction = sgd.predict(X)
            predicted_[label] = prediction
        output = []
        for _, row in predicted_.iterrows():
            not_predicted=True
            for label in self.labels_:
                if row[label] == 1:
                    output.append(label)
                    not_predicted=False
                    break
            if not_predicted:
                output.append(None)

        return output

    def get_accuracy(self, predicted, label):
        count = 0
        for index in range(len(label)):
            if predicted[index] == label[index]:
                count = count + 1
        return count / len(label)
