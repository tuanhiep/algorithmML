#  Copyright (c) 2020. Tuan-Hiep TRAN

# use linkage of Scipy

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt


class MyLinkage:
    def __init__(self, X, method, metric):
        self.final_clusters = None
        self.row_cluster = linkage(X, method=method, metric=metric)
        # print(self.row_cluster)

    def get_final_cluster(self, maxclust):
        self.final_clusters = fcluster(self.row_cluster, maxclust, criterion='maxclust')
        print("Final Cluster: ")
        print(self.final_clusters)
        return self.final_clusters

    def plot_dendrogram(self):
        dendrogram(self.row_cluster, labels=self.final_clusters)
        plt.tight_layout()
        plt.ylabel('Euclidean distance')
        plt.title("Dendrogram")
        plt.show()

