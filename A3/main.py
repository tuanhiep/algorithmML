#  Copyright (c) 2020. Tuan Hiep TRAN
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from A3.decision_tree import MyDecisionTree
from A3.k_nearest_neighbors import MyKNeighborsClassifier
from A3.perceptron import MyPerceptron
from A3.svm import MySVM
import pandas as pd
import sys
import time

# Get the parameters from command line input
classifier_name = sys.argv[1].lower()
data_source = sys.argv[2]
argv3 = sys.argv[3]

# Select and initialize the corresponding classifier
if classifier_name == "perceptron":
    try:
        argv4 = sys.argv[4]
        learning_rate = float(argv3)
        number_iteration = int(argv4)
    except ValueError:
        print(" The fourth argument is learning rate and should be of float type, the fifth argument is number of "
              "iteration and should be of integer type")

    classifier = MyPerceptron(learning_rate, number_iteration)
elif classifier_name == "decision_tree":
    try:
        criterion = argv3
        argv4 = sys.argv[4]
        if criterion.lower() != "gini" and criterion.lower() != "entropy":
            raise ValueError("Criterion error !")
        random_state = int(argv4)
    except ValueError:
        print(" The fourth argument is criterion of decision tree and should be gini or entropy, the fifth argument"
              " is random state and should be of integer type")

    classifier = MyDecisionTree(criterion, random_state)
elif classifier_name == "svm":
    kernel = argv3
    classifier = MySVM(kernel)
elif classifier_name == "knn":
    try:
        k_neighbor = int(argv3)
    except ValueError:
        print("The fourth argument is the number of k-nearest neighbors and should be an integer")
    classifier = MyKNeighborsClassifier(k_neighbor)
else:
    # error checking for classifier name which is entered in command line
    raise ValueError(
        'Classifier name is not correct ! Please enter : 1. perceptron 2. decision_tree 3. svm 4.knn')
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
    df = pd.read_csv(data_source, header=None)
    # select labels
    Y = df.iloc[:, -1].values
    # extract features
    X = df.iloc[:, 0:-1].values

    # split training set and testing set of time series data
    tscv = TimeSeriesSplit( n_splits=2,max_train_size=None)
    # count = 1
    # print("Split time seris data into following partitions: ")
    for train_index, test_index in tscv.split(X):
        # print("For partition: " + str(count))
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        break
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
