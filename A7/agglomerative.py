#  Copyright (c) 2020. Tuan-Hiep TRAN

from sklearn.cluster import AgglomerativeClustering


class MyAgglomerativeClustering:
    def __init__(self, n_cluster, metric, linkage):
        self.cluster = AgglomerativeClustering(n_clusters=n_cluster,
                                               affinity=metric,
                                               linkage=linkage)

    def fit_predict(self, X):
        cluster_labels = self.cluster.fit_predict(X)
        print(cluster_labels)
        return cluster_labels
