#  Copyright (c) 2020. Tuan-Hiep TRAN
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:

    def __init__(self, n_estimators, learning_rate):
        self.baseline = DecisionTreeClassifier(criterion='entropy',
                                               max_depth=1,
                                               random_state=1)
        self.ada = AdaBoostClassifier(base_estimator=self.baseline,
                                      n_estimators=n_estimators,
                                      learning_rate=learning_rate,
                                      random_state=1)

    def fit(self, X, Y):
        self.ada.fit(X, Y)

    def baseline_fit(self, X, Y):
        self.baseline.fit(X, Y)

    def predict(self, X):
        return self.ada.predict(X)

    def score(self, X, Y):
        return self.ada.score(X, Y)

    def baseline_score(self, X, Y):
        return self.baseline.score(X, Y)
