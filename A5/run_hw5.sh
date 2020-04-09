#!/bin/bash
#prepare the minist subset data set
echo "Preparing mnist_subset dataset... "
python -W ignore prepare_mnist_subset.py ../csv/mnist_subset.csv

#Start the experiments
echo "Begin:"

#IRIS DATA SET
dataset="../csv/iris.csv"
echo "Do experiment with $dataset data set: "
for n_components in 2 3 4; do
  echo "NUMBER OF COMPONENTS : $n_components"
  for analyser in "pca" "lda" "kpca"; do
    echo "ANALYSER: $analyser"
    if [ $analyser = "kpca" ]; then
      python -W ignore main.py -data $dataset -n $n_components "-${analyser}" -k "rbf" -g 15 -md 10
    else
      python -W ignore main.py -data $dataset -n $n_components "-${analyser}" -md 10
    fi
  done
done

#MNIST SUBSET DATA SET
dataset="../csv/mnist_subset.csv"
echo "Do experiment with $dataset data set: "
for n_components in 300 600 784; do
  echo "NUMBER OF COMPONENTS : $n_components"
  for analyser in "pca" "lda" "kpca"; do
    echo "ANALYSER: $analyser"
    if [ $analyser = "kpca" ]; then
      python -W ignore main.py -data $dataset -n $n_components "-${analyser}" -k "rbf" -g 15 -md 500
    else
      python -W ignore main.py -data $dataset -n $n_components "-${analyser}" -md 500
    fi
  done
done

echo "End"
