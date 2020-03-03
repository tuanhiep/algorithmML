#  Copyright (c) 2020. Tuan Hiep TRAN
from sklearn.tree import DecisionTreeClassifier


class MyDecisionTree:
    def __init__(self, criterion, random_state):
        self.decision_tree = DecisionTreeClassifier(criterion=criterion, random_state=random_state)

    def fit(self, X, y):
        self.decision_tree.fit(X, y)

    def predict(self, X):
        return self.decision_tree.predict(X)

    def score(self, X, y):
        return self.decision_tree.score(X, y)
