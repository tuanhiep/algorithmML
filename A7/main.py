#  Copyright (c) 2020. Tuan-Hiep TRAN
import argparse

from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from A7.agglomerative import MyAgglomerativeClustering
from A7.dbscan import MyDBScan
from A7.kmeans import MyKmeans
from os import path
import time
import pandas as pd
from A7.linkage import MyLinkage
from sklearn import metrics


# Extrinsic Measure: Get the Adjusted Rand Score for clustering quality
def get_adjusted_rand_score(clusterIds, ground_truth):
    score = metrics.adjusted_rand_score(clusterIds, ground_truth.flatten())
    return score


# Intrinsic Measure: Get the Silhouette Score for clustering quality
def get_silhouette_score(X, pred_labels, metric):
    score = silhouette_score(X, pred_labels.flatten(), metric=metric)
    return score


parser = argparse.ArgumentParser(description="Regression")
parser.add_argument("-data", "--dataSet", type=str, required=True, help="the path to considered data set")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-kmeans", "--kmeans", action="store_true", help="Kmeans  ")
group.add_argument("-k_elbow", "--k_elbow", action="store_true", help="Call Elbow for Kmeans  ")
group.add_argument("-linkage", "--linkage", action="store_true", help="Linkage -hierarchical clustering ")
group.add_argument("-agglomerative", "--agglomerative", action="store_true", help="Agglomerative clustering")
group.add_argument("-dbscan", "--dbscan", action="store_true", help="DBScan")
group.add_argument("-dbscan_determine_parameters", "--dbscan_determine_parameters", action="store_true", help="DBScan")
parser.add_argument("-n_cluster", "--n_cluster", type=int, help="n_cluster")
parser.add_argument("-n_init", "--n_init", type=int, help="n_init")
parser.add_argument("-max_iter", "--max_iter", type=int, help="max_iter")
parser.add_argument("-metric", "--metric", type=str, help="metric of linkage")
parser.add_argument("-method_linkage", "--method_linkage", type=str, help="method of linkage")
parser.add_argument("-eps", "--eps", type=float, help="eps of DBScan")
parser.add_argument("-min_samples", "--min_samples", type=int, help="min_samples of DBScan")
parser.add_argument("-min_n_cluster", "--min_n_cluster", type=int, help="min_n_cluster")
parser.add_argument("-max_n_cluster", "--max_n_cluster", type=int, help="max_n_cluster")
parser.add_argument("-threshold_noise", "--threshold_noise", type=float, help="threshold_noise tolerated in DBScan")
args = parser.parse_args()

# check arguments
if args.kmeans and (args.n_cluster is None or args.n_init is None or args.max_iter is None):
    parser.error("--kmeans requires --n_cluster and --n_init and --max_iter")
if args.k_elbow and (
        args.min_n_cluster is None or args.max_n_cluster is None or args.n_init is None or args.max_iter is None):
    parser.error("--k_elbow requires --min_n_cluster and --max_n_cluster and --n_init and --max_iter")
elif args.linkage and (args.metric is None or args.method_linkage is None or args.max_n_cluster is None):
    parser.error("--linkage requires --metric and --method and --max_n_cluster")
elif args.agglomerative and (args.n_cluster is None or args.metric is None or args.linkage is None):
    parser.error("--agglomerative requires --n_cluster and --metric and --method_linkage")
elif args.dbscan and (args.eps is None or args.min_samples is None or args.metric is None):
    parser.error("--dbscan requires --eps and --min_samples and --metric")

# load data set
if path.exists(args.dataSet):
    df = pd.read_csv(args.dataSet, header=None)
    # select labels
    Y = df.iloc[:, -1].values.reshape(-1, 1)
    # extract features
    X = df.iloc[:, 0:-1].values
else:
    raise ValueError("Data set not found !")
# Start to measure running time of training process
start_time = time.time()
if __name__ == "__main__":
    if args.kmeans:
        cluster = MyKmeans(args.n_init, args.max_iter)
        clusterIds = cluster.fit_predict(X, args.n_cluster)
        print("Time for processing is  %s seconds" % (time.time() - start_time))
        print("Extrinsic Measure - Adjusted Rand score=%s" % get_adjusted_rand_score(clusterIds, Y))
        print("Intrinsic Measure - Silhouette Score score=%s" % get_silhouette_score(X, clusterIds, "euclidean"))
    if args.k_elbow:
        cluster = MyKmeans(args.n_init, args.max_iter)
        cluster.elbow(X, args.min_n_cluster, args.max_n_cluster)
    elif args.linkage:
        cluster = MyLinkage(X, args.method_linkage, args.metric)
        clusterIds = cluster.get_final_cluster(args.max_n_cluster)
        print("Time for processing is  %s seconds" % (time.time() - start_time))
        print("Extrinsic Measure - Adjusted Rand score=%s" % get_adjusted_rand_score(clusterIds, Y))
        print("Intrinsic Measure - Silhouette Score score=%s" % get_silhouette_score(X, clusterIds, args.metric))

        cluster.plot_dendrogram()
    elif args.agglomerative:
        cluster = MyAgglomerativeClustering(args.n_cluster, args.metric, args.method_linkage)
        clusterIds = cluster.fit_predict(X)
        print("Time for processing is  %s seconds" % (time.time() - start_time))
        print("Extrinsic Measure - Adjusted Rand score=%s" % get_adjusted_rand_score(clusterIds, Y))
        print("Intrinsic Measure - Silhouette Score score=%s" % get_silhouette_score(X, clusterIds, args.metric))

    elif args.dbscan:
        cluster = MyDBScan(args.eps, args.min_samples, args.metric)
        clusterIds = cluster.fit_predict(X)
        print("Time for processing is  %s seconds" % (time.time() - start_time))
        print("Extrinsic Measure - Adjusted Rand score=%s" % get_adjusted_rand_score(clusterIds, Y))
        print("Intrinsic Measure - Silhouette Score score=%s" % get_silhouette_score(X, clusterIds, args.metric))

    elif args.dbscan_determine_parameters:
        MyDBScan.determine_parameter(X, args.threshold_noise)
