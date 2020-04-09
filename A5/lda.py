#  Copyright (c) 2020. Tuan-Hiep TRAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class myLDA:
    def __init__(self, n_components):
        self.lda = LDA(n_components=n_components)

    def fit_transform(self, X, Y):
        return self.lda.fit_transform(X, Y)

    def transform(self, X):
        return self.lda.transform(X)
