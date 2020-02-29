import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from A2.adaline import Adaline
from A2.one_vs_rest import OneVsRest
from A2.perceptron import Perceptron
from A2.sgd import StochasticGD
import time

# Get the parameters from command line input
classifier_name = sys.argv[1].lower()
data_path = sys.argv[2]
learning_rate = float(sys.argv[3])
number_iteration = int(sys.argv[4])
# Select and initialize the corresponding classifier
if classifier_name == "perceptron":
    classifier = Perceptron(eta=learning_rate, n_iter=number_iteration)
elif classifier_name == "adaline":
    classifier = Adaline(eta=learning_rate, n_iter=number_iteration)
elif classifier_name == "sgd":
    classifier = StochasticGD(eta=learning_rate, n_iter=number_iteration)
elif classifier_name == "one_vs_rest":
    classifier = OneVsRest(eta=learning_rate, n_iter=number_iteration)
else:
    # error checking for classifier name which is entered in command line
    raise ValueError('Classifier name is not correct ! Please enter : 1. perceptron 2. adaline 3. sgd 4. one_vs_rest')
# read data from dataset

df = pd.read_csv(data_path, header=None)
if classifier_name in ["perceptron", "adaline", "sgd"]:
    print("Classifier %s" % classifier_name)
    Y = df.iloc[:, -1].values  # select labels
    # Convert the class labels to two integer, the first class label will be -1, otherwise 1
    Y = np.where(Y == Y[0], 1, -1)
    # extract features
    X = df.iloc[:, 0:-1].values
    # Create training set and test set
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)
    # Because we don't split data set for this homework
    X_train = X
    X_test = X
    Y_train = Y
    Y_test = Y
    # Start to measure running time of trainning process
    start_time = time.time()
    # training
    classifier.fit(X_train, Y_train)
    print("Time for training is  %s seconds" % (time.time() - start_time))
    # plot the result
    plt.plot(range(1, len(classifier.cost_) + 1), classifier.cost_, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Cost [number of errors in case Perceptron]')
    plt.show()
    # predict
    prediction = classifier.predict(X_test)
    # print(prediction)
    print("The accuracy: " + str(classifier.get_accuracy(prediction, Y_test)))
elif classifier_name == "one_vs_rest":
    print("Classifier %s" % classifier_name)
    # select labels
    Y = df.iloc[:, -1].values
    # extract features
    X = df.iloc[:, 0:-1].values
    # Create training set and test set
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)
    # Because we don't split data set for this homework
    X_train = X
    X_test = X
    Y_train = Y
    Y_test = Y
    # Start to measure running time of trainning process
    start_time = time.time()
    # training
    classifier.fit(X_train, Y_train)
    print("Time for training is  %s seconds" % (time.time() - start_time))

    # predict
    prediction = classifier.predict(X_test)
    # print(prediction)
    print("The accuracy: " + str(classifier.get_accuracy(prediction, Y_test)))
