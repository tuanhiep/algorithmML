#  Copyright (c) 2020. Tuan-Hiep TRAN
from sklearn.decomposition import PCA


class myPCA:

    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)

    def fit_transform(self, X, Y):
        return self.pca.fit_transform(X, Y)

    def transform(self, X):
        return self.pca.transform(X)

