Please follow these following steps to run my program:

1. Prepare data
The data sets for this experiment is in the folder /csv. With iris.csv we don't need to do any additional preparing process.
However for the second data set: breast-cancer-wisconsin.data from: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
,additional preparing process need to be done because the data contains some missing value.

Go to folder A2, run this command:

# python prepare_data.py csv/breast-cancer-wisconsin.data csv/breast-cancer-wisconsin.csv

This command will create new file breast-cancer-wisconsin.csv in folder /csv

2. Run the program

Go to the folder A2, do this command:

python main.py [name_of_classifier] [path_to_data_set] [learning_rate] [number_iteration]


+ name_of_classifer can be: perceptron, adaline, sgd, one_vs_rest
+ path_to_data_set can be:
            csv/breast-cancer-wisconsin.csv
 or         csv/iris.csv
+ learning_rate can be: 0.0001
+ number_iteration can be: 100

 Example:

- To do the experiment with Stochastic Gradient Descent classifier for iris data set, the command is:

#  python main.py sgd csv/iris.csv 0.0001 100

- To do the experiment with Stochastic Gradient Descent classifier for breast-cancer-wisconsin data set, the command is:

#  python main.py sgd csv/breast-cancer-wisconsin.csv 0.0001 100

- To do the experiment with Adaline classifier for Iris data set, the command is:

#  python main.py adaline csv/iris.csv 0.0001 100

- To do the experiment with Adaline classifier for breast-cancer-wisconsin data set, the command is:

#  python main.py adaline csv/breast-cancer-wisconsin.csv 0.0001 100

- To do the experiment with Perceptron classifier for Iris data set, the command is:

#  python main.py perceptron csv/iris.csv 0.0001 100

- To do the experiment with Perceptron classifier for breast-cancer-wisconsin data set, the command is:

#  python main.py perceptron csv/breast-cancer-wisconsin.csv 0.0001 100

- To do the experiment with multiple class one vs rest classifier for Iris data set, the command is:

#  python main.py one_vs_rest csv/iris.csv 0.0001 100

- To do the experiment with multiple class one vs rest classifier for breast-cancer-wisconsin data set, the command is:

#  python main.py one_vs_rest csv/breast-cancer-wisconsin.csv 0.0001 100



