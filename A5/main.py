#  Copyright (c) 2020. Tuan-Hiep TRAN

import argparse
from os import path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from A5.kpca import myKernelPCA
from A5.lda import myLDA
from A5.pca import myPCA
# from kpca import myKernelPCA
# from lda import myLDA
# from pca import myPCA
from sklearn.datasets import fetch_openml


# function to print the performance metrics
def print_performance_metrics(y_predicted, y):
    acc = accuracy_score(y_predicted, y)
    print("Accuracy = ", acc)
    print('Precision: %.3f' % precision_score(y_true=y, y_pred=y_predicted, average='micro'))
    print('Recall: %.3f' % recall_score(y_true=y, y_pred=y_predicted, average='micro'))
    print('F1: %.3f' % f1_score(y_true=y, y_pred=y_predicted, average='micro'))


# parse the arguments from command line
parser = argparse.ArgumentParser(description="Dimensionality Reduction")
parser.add_argument("-n", "--numberComponent", type=int, required=True, help="number of components for this analysis")
parser.add_argument("-data", "--dataSet", type=str, required=True, help="name of the data set for this program")
parser.add_argument("-cv", "--cv", type=int, default=6, help="number of cross validation folds")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-pca", "--pca", action="store_true", help="use PCA ")
group.add_argument("-lda", "--lda", action="store_true", help="use LDA ")
group.add_argument("-kpca", "--kpca", action="store_true", help="use KPCA ")
parser.add_argument("-k", "--kernel", type=str, help="kernel method")
parser.add_argument("-g", "--gamma", type=int, help="gamma parameter")
parser.add_argument("-md", "--max_depth", type=int, help="max depth of baseline decision tree parameter")
args = parser.parse_args()

if args.kpca and (args.kernel is None or args.gamma is None):
    parser.error("--kpca requires --kernel and --gamma")

if __name__ == "__main__":
    if args.pca:
        analyser = myPCA(n_components=args.numberComponent)
    elif args.lda:
        analyser = myLDA(n_components=args.numberComponent)
    elif args.kpca:
        analyser = myKernelPCA(n_components=args.numberComponent, kernel=args.kernel, gamma=args.gamma)

# load data set
if path.exists(args.dataSet):
    df = pd.read_csv(args.dataSet, header=None)
    # select labels
    Y = df.iloc[:, -1].values
    # extract features
    X = df.iloc[:, 0:-1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)
else:
    raise ValueError("Data set not found !")

# Standardize the features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
# Realize the dimensionality reduction

# transform from X space to Z space
X_train_z_space = analyser.fit_transform(X_train_std, Y_train)
X_test_z_space = analyser.transform(X_test_std)

# Get the result
print("DECISION TREE : ")
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=1)
# in X space

# Use cross validation for training set
Y_predicted_training = cross_val_predict(tree_model, X_train, Y_train, cv=args.cv)
print("Training set result:")
print_performance_metrics(Y_predicted_training, Y_train)
# For testing set
tree_model.fit(X_train, Y_train)
Y_predicted_testing = tree_model.predict(X_test)
print("Testing set result:")
print_performance_metrics(Y_predicted_testing, Y_test)

# in Z space

print("DECISION TREE + DIMENSIONALITY REDUCTION : ")

# Use cross validation for training set
Y_predicted_training_z_space = cross_val_predict(tree_model, X_train_z_space, Y_train, cv=args.cv)
print("Training set result:")
print_performance_metrics(Y_predicted_training_z_space, Y_train)
# For testing set
tree_model.fit(X_train_z_space, Y_train)
Y_predicted_testing_z_space = tree_model.predict(X_test_z_space)
print("Testing set result:")
print_performance_metrics(Y_predicted_testing_z_space, Y_test)
