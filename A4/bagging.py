#  Copyright (c) 2020. Tuan-Hiep TRAN
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


class Bagging:

    def __init__(self, n_estimators, max_samples, max_features, bootstrap):
        self.baseline = DecisionTreeClassifier(criterion='entropy',
                                               max_depth=None,
                                               random_state=1)
        self.bag = BaggingClassifier(base_estimator=self.baseline,
                                     n_estimators=n_estimators,
                                     max_samples=max_samples,
                                     max_features=max_features,
                                     bootstrap=bootstrap)

    def fit(self, X, Y):
        self.bag.fit(X, Y)

    def baseline_fit(self, X, Y):
        self.baseline.fit(X, Y)

    def predict(self, X):
        return self.bag.predict(X)

    def score(self, X, Y):
        return self.bag.score(X, Y)

    def baseline_score(self, X, Y):
        return self.baseline.score(X, Y)
