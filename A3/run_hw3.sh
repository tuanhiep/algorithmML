#!/bin/bash
# Prepare second datasets
echo "Preparing second datasets..."
python -W ignore prepare_data.py csv/datatraining.txt csv/data-clean.csv
# Run perceptron with digit data set
echo "Do experiment with digits data set: "
echo "PERCEPTRON: learning rate=0.001, number of iteration =100"
python -W ignore main.py perceptron digit 0.001 100
# Run decision tree classifier with digit data set and gini index
echo "DECISION TREE: index= gini, random state =0"
python -W ignore main.py decision_tree digit gini 0
# Run decision tree classifier with digit data set and entropy index
echo "DECISION TREE: index= entropy, random state =0"
python -W ignore main.py decision_tree digit entropy 0
# Run Support Vector Machine classifier with kernel= linear
echo "SVM: kernel= linear"
python -W ignore main.py svm digit linear
# Run Support Vector Machine classifier with kernel= rbf
echo "SVM: kernel= rbf"
python -W ignore main.py svm digit rbf
# Run K-nearest neighbors classifier with digit data set
echo "KNN:"

for k in 10 20 30
do
  echo "Run K-nearest neighbors classifier with k= ${k}"
  python -W ignore main.py knn digit "${k}"
done

# Run perceptron with second data set
echo "Do experiment with second data set: "
echo "PERCEPTRON: learning rate=0.001, number of iteration =100"
python -W ignore main.py perceptron csv/data-clean.csv 0.001 100
# Run decision tree classifier with digit data set and gini index
echo "DECISION TREE: index= gini, random state =0"
python -W ignore main.py decision_tree csv/data-clean.csv gini 0
# Run decision tree classifier with digit data set and entropy index
echo "DECISION TREE: index= entropy, random state =0"
python -W ignore main.py decision_tree csv/data-clean.csv entropy 0
# Run Support Vector Machine classifier with kernel= linear
echo "SVM: kernel= linear"
python -W ignore main.py svm csv/data-clean.csv linear
# Run Support Vector Machine classifier with kernel= rbf
echo "SVM: kernel= rbf"
python -W ignore main.py svm csv/data-clean.csv rbf
# Run K-nearest neighbors classifier with digit data set

echo "KNN:"
for k in 10 20 30
do
  echo "Run K-nearest neighbors classifier with k= ${k}"
  python -W ignore main.py knn csv/data-clean.csv "${k}"
done
