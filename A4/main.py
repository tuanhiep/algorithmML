#  Copyright (c) 2020. Tuan-Hiep TRAN
import sys
import time
from os import path

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Get the parameters from command line input
from A4.adaboost import AdaBoost
from A4.bagging import Bagging
from A4.random_forest import RandomForest

classifier_name = sys.argv[1].lower()
data_source = sys.argv[2]

# Select and initialize the corresponding classifier
if classifier_name == "random_forest":
    try:
        n_estimators = int(sys.argv[3])
        criterion = sys.argv[4]
        max_depth = int(sys.argv[5])
        min_samples_split = int(sys.argv[6])
        min_samples_leaf = int(sys.argv[7])
    except ValueError:
        print(" The arguments must be: n_estimators(integer) criterion(gini or entropy) max_depth(integer)"
              " min_samples_split(integer) min_samples_leaf(integer)  ")
    classifier = RandomForest(n_estimators=n_estimators,
                              criterion=criterion,
                              max_depth=max_depth,
                              min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf)
elif classifier_name == "bagging":
    try:
        n_estimators = int(sys.argv[3])
        max_samples = int(sys.argv[4])
        max_features = int(sys.argv[5])
        bootstrap = int(sys.argv[6])
    except ValueError:
        print("The arguments must be: n_estimators(integer) max_samples(integer) max_features(integer)"
              " bootstrap(integer)")
    classifier = Bagging(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features,
                         bootstrap=bootstrap)
elif classifier_name == "adaboost":
    try:
        n_estimators = int(sys.argv[3])
        learning_rate = float(sys.argv[4])
    except ValueError:
        print("The arguments must be: n_estimators(integer) learning_rate(float) ")
    classifier = AdaBoost(n_estimators=n_estimators, learning_rate=learning_rate)

else:
    # error checking for classifier name which is entered in command line
    raise ValueError(
        'Classifier name is not correct ! Please enter : 1. random_forest 2. bagging 3. adaboost')
# Load data set
if data_source == "digit":
    # Load the digits data set
    digits = datasets.load_digits()
    # Create our X and y data
    Y = digits.target.reshape(-1, 1)
    X = digits.data
    # split training and testing data set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)

else:
    if path.exists(data_source):
        df = pd.read_csv(data_source, header=None)
        # select labels
        Y = df.iloc[:, -1].values
        # extract features
        X = df.iloc[:, 0:-1].values
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)
    else:
        raise ValueError("Data set not found !")

# Normalize data before training, it's necessary when we have time series data type
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Start the experiment
print("Classifier: " + classifier_name)
# Start to measure running time of training process
start_time = time.time()
# training
classifier.fit(X_train, Y_train)
print("Time for training is  %s seconds" % (time.time() - start_time))

# print out the accuracy of training data set
predicted_Y_train = classifier.predict(X_train)
print("The accuracy for training set: " + str(classifier.score(X_train, Y_train)))

# print out the accuracy of testing data set
predicted_Y_test = classifier.predict(X_test)
print("The accuracy for testing set: " + str(classifier.score(X_test, Y_test)))

# print out the accuracy of baseline methods:

# training baseline method
classifier.baseline_fit(X_train, Y_train)

predicted_Y_train = classifier.predict(X_train)
print("The accuracy of baseline method for training set: " + str(classifier.baseline_score(X_train, Y_train)))

predicted_Y_test = classifier.predict(X_test)
print("The accuracy of baseline method for testing set: " + str(classifier.baseline_score(X_test, Y_test)))
