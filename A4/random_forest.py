#  Copyright (c) 2020. Tuan-Hiep TRAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class RandomForest:

    def __init__(self,
                 n_estimators,
                 criterion,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf):
        self.random_forest_classifier = RandomForestClassifier(n_estimators=n_estimators,
                                                               criterion=criterion,
                                                               max_depth=max_depth,
                                                               min_samples_split=min_samples_split,
                                                               min_samples_leaf=min_samples_leaf)
        self.baseline = DecisionTreeClassifier(criterion=criterion, random_state=0)

    def fit(self, X, Y):
        self.random_forest_classifier.fit(X, Y)
        self.baseline.fit(X, Y)

    def baseline_fit(self, X, Y):
        self.baseline.fit(X, Y)

    def predict(self, X):
        return self.random_forest_classifier.predict(X)

    def score(self, X, Y):
        return self.random_forest_classifier.score(X, Y)

    def baseline_score(self, X, Y):
        return self.baseline.score(X, Y)
