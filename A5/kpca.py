#  Copyright (c) 2020. Tuan-Hiep TRAN
from sklearn.decomposition import KernelPCA


class myKernelPCA:
    def __init__(self, n_components, kernel, gamma):
        self.kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)

    def fit_transform(self, X, Y):
       return self.kpca.fit_transform(X, Y)

    def transform(self, X):
       return self.kpca.transform(X)
