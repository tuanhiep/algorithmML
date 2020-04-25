#!/bin/bash
#Start the experiments
echo "Starting..."

# Prepare second datasets
echo "Preparing second datasets..."
python -W ignore prepare_dataset.py csv/faults.csv csv/faults_fine.csv
echo "DOING EXPERIMENTS:"
# IRIS DATA SET
echo "HOUSING DATA SET"
data_set="csv/iris.csv"

# using the K-means algorithm offered by Scikit-learn library
python -W ignore main.py -data $dataset -kmeans -n_cluster 3 -n_init 10 -max_iter 300
python -W ignore main.py -data $dataset -k_elbow -min_n_cluster 1 -max_n_cluster 30 -n_init 10 -max_iter 300

# using a hierarchical approach offered by SciPy library
python -W ignore main.py -data $dataset -linkage -method "complete" -metric "euclidean" -max_n_cluster 3

# using a hierarchical approach offered by Scikit-learn library

python -W ignore main.py -data $dataset -agglomerative -n_cluster 3 -method "complete" -metric "euclidean"

# using the DBSCAN density based method offered by Scikit-learn library.
python -W ignore main.py -data $dataset -dbscan -eps 0.4 -min_samples 5 -metric "euclidean"
python -W ignore main.py -data $dataset -dbscan_determine_parameters -threshold_noise 0.05

# FAULTY STEEL PLATES DATA SET
echo "FAULTY STEEL PLATES DATA SET"
data_set="csv/faults_fine.csv"


# using the K-means algorithm offered by Scikit-learn library
python -W ignore main.py -data $dataset -kmeans -n_cluster 3 -n_init 10 -max_iter 300
python -W ignore main.py -data $dataset -k_elbow -min_n_cluster 1 -max_n_cluster 30 -n_init 10 -max_iter 300

# using a hierarchical approach offered by SciPy library
python -W ignore main.py -data $dataset -linkage -method "complete" -metric "euclidean"

# using a hierarchical approach offered by Scikit-learn library

python -W ignore main.py -data $dataset -agglomerative -n_cluster 3 -method "complete" -metric "euclidean"

# using the DBSCAN density based method offered by Scikit-learn library.
python -W ignore main.py -data $dataset -dbscan -eps 0.4 -min_samples 5 -metric "euclidean"
python -W ignore main.py -data $dataset -dbscan_determine_parameters -threshold_noise 0.05

echo "End !"