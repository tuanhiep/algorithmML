#!/bin/bash
# Prepare second datasets
echo "Preparing second datasets..."
python -W ignore prepare_data.py csv/mammographic_masses.data csv/mammographic_masses-clean.csv
# Run perceptron with digit data set
for dataset in "digit" "csv/mammographic_masses-clean.csv"; do
  echo "Do experiment with $dataset data set: "

  #Run Random Forest
  echo "RANDOM FOREST:"

  #parameters: n_estimators, criterion, max_depth, min_samples, min_samples_leaf

  # Define the arrays
  n_estimators=(25 50 75)
  criterion=("gini" "gini" "gini")
  max_depth=(25 25 25)
  min_samples=(2 2 2)
  min_samples_leaf=(1 1 1)
  # get the length of the arrays
  length=${#n_estimators[@]}
  # do the loop
  for ((i = 0; i < length; i++)); do
    echo "n_estimators=${n_estimators[$i]} criterion=${criterion[$i]} max_depth=${max_depth[$i]} min_samples=${min_samples[$i]} min_samples_leaf=${min_samples_leaf[$i]} "
    python -W ignore main.py random_forest $dataset "${n_estimators[$i]}" "${criterion[$i]}" "${max_depth[$i]}" "${min_samples[$i]}" "${min_samples_leaf[$i]}"

  done

  #Run Bagging
  echo "BAGGING:"

  #parameters: min_samples, max_features, bootstrap

  # Define the arrays
  n_estimators=(25 50 75)
  max_samples=(1 1 1)
  max_features=(1 1 1)
  bootstrap=(1 1 1)
  # get the length of the arrays
  length=${#min_samples[@]}
  # do the loop
  for ((i = 0; i < length; i++)); do
    echo "n_estimators = ${n_estimators[$i]} max_samples=${min_samples[$i]} max_features=${max_features[$i]} bootstrap=${bootstrap[$i]} "
    python -W ignore main.py bagging $dataset  "${n_estimators[$i]}" "${max_samples[$i]}" "${max_features[$i]}" "${bootstrap[$i]}"
  done

  #Run AdaBoost
  echo "ADABOOST"

  #parameters: n_estimators, learning_rate

  # Define the arrays
  n_estimators=(250 500 750)
  learning_rate=(0.1 0.1 0.1)
  # get the length of the arrays
  length=${#n_estimators[@]}
  # do the loop
  for ((i = 0; i < length; i++)); do
    echo "n_estimators=${n_estimators[$i]} learning_rate=${learning_rate[$i]} "
    python -W ignore main.py adaboost $dataset "${n_estimators[$i]}" "${learning_rate[$i]}"

  done

done
