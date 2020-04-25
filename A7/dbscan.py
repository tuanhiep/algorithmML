#  Copyright (c) 2020. Tuan-HiepTRAN

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np


# threshold_noise is maximum percentage of noise which is tolerated
def get_thresold_kdist(distanceKi, thresold_noise):
    return distanceKi[round(len(distanceKi) * thresold_noise)]


class MyDBScan:
    def __init__(self, eps, min_samples, metric):
        self.db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

    def fit_predict(self, X):
        y_db = self.db.fit_predict(X)
        print(y_db)  # shows the cluster id of each point
        return y_db
    @staticmethod
    def determine_parameter(X, threshold_noise):
        K = len(X) - 1
        n = len(X)
        nn = NearestNeighbors(n_neighbors=(K + 1))
        nbrs = nn.fit(X)
        # indices and distances to k-nearest neighbors
        distances, indices = nbrs.kneighbors(X)
        # distanceK store the k-dist for all the points, each row is a point, the value is k-dist
        distanceK = np.empty([K, n])
        thresholdK = np.zeros(shape=(K, 1))
        # i start from zero so we need to plus 1 while constructing k-dist
        for i in range(K):
            distance_Ki = distances[:, (i + 1)]
            distance_Ki.sort()
            distance_Ki = distance_Ki[::-1]
            distanceK[i] = distance_Ki
            # figure out the k-dist so that we don't have more than threshold % of noise
            threshold_k_dist = get_thresold_kdist(distanceK[i], threshold_noise)
            thresholdK[i] = threshold_k_dist
        # print(distanceK)
        print("To have noise less than {} we should choose one of the parameters :".format(threshold_noise))
        for i in range(5):
            print("MinPts= {} and Eps= {}".format(i + 1, thresholdK[i]))
        # print the plot for visualization
        for i in range(5):
            plt.plot(distanceK[i], label="K=%d" % (i + 1))
        plt.ylabel("distance")
        plt.xlabel("points")
        plt.title("K-dist graph- 5 first examples")
        plt.legend()
        plt.show()
