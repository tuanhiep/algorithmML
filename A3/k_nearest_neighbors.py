#  Copyright (c) 2020. Tuan Hiep TRAN
from sklearn.neighbors import KNeighborsClassifier


class MyKNeighborsClassifier:
    def __init__(self, k_neighbor):
        self.k_neighbors_classifier = KNeighborsClassifier(n_neighbors=k_neighbor)

    def fit(self, X, y):
        self.k_neighbors_classifier.fit(X, y)

    def predict(self, X):
        return self.k_neighbors_classifier.predict(X)

    def score(self, X, y):
        return self.k_neighbors_classifier.score(X, y)
