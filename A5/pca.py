#  Copyright (c) 2020. Tuan-Hiep TRAN
from sklearn.decomposition import PCA


class myPCA:

    def __init__(self, nComponent):
        self.pca = PCA(n_component=nComponent)
