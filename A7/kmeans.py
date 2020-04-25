#  Copyright (c) 2020. Tuan-Hiep TRAN

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class MyKmeans:
    def __init__(self, n_init, max_iter):
        self.n_init = n_init
        self.max_iter = max_iter
        self.name = "Kmeans"

    def fit_predict(self, X, n_cluster):
        self.model = KMeans(n_clusters= n_cluster,
                            init='random',
                            n_init=self.n_init,
                            max_iter=self.max_iter,
                            tol=1e-04,
                            random_state=0)
        return self.model.fit_predict(X)

    def get_centers(self):
        return self.model.cluster_centers_

    def get_inertia(self):
        return self.model.inertia_

    def elbow(self, X, min_n_cluster, max_n_cluster):
        # Elbow method
        distortions = []
        # Calculate distortions
        for i in range(min_n_cluster, max_n_cluster):
            km = KMeans(n_clusters=i,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        random_state=0)
            km.fit(X)
            distortions.append(km.inertia_)
        # Plot distortions for different K
        plt.plot(range(min_n_cluster, max_n_cluster), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion- Sum Squared Error (SSE)')
        plt.title("Kmeans - Elbow Method")
        plt.tight_layout()
        plt.show()
