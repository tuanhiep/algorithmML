#  Copyright (c) 2020. Tuan-Hiep TRAN

# parse the arguments from command line
import argparse
import time
from os import path

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from A6.lasso import MyLasso
from A6.linear_regression import MyLinearRegression
from A6.linear_regression_normal_equation import MyLinearRegressionNE
from A6.non_linear_regression import MyDecisionTreeRegressor
from A6.ransac_regressor import MyRANSACRegressor
from A6.ridge import MyRidge

parser = argparse.ArgumentParser(description="Regression")
parser.add_argument("-data", "--dataSet", type=str, required=True, help="the path to considered data set")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-lr", "--lr", action="store_true", help="Linear Regression  ")
group.add_argument("-lrne", "--lrne", action="store_true", help="Linear Regression Normal Equation ")
group.add_argument("-lasso", "--lasso", action="store_true", help="LASSO")
group.add_argument("-ridge", "--ridge", action="store_true", help="RIDGE")
group.add_argument("-nlr", "--nlr", action="store_true", help="Non linear regression - Decision Tree Regressor")
group.add_argument("-ransac", "--ransac", action="store_true", help="RANSAC")
parser.add_argument("-alpha", "--alpha", type=float, help="alpha parameter of LASSO")
parser.add_argument("-min_sample", "--min_sample", type=float, help="min_sample parameter of RANSAC")
parser.add_argument("-max_depth", "--max_depth", type=float, help="max_depth parameter of Decision Tree Regressor")

args = parser.parse_args()

if args.lasso and args.alpha is None:
    parser.error("--lasso requires --alpha")
elif args.ridge and args.alpha is None:
    parser.error("--ridge requires --alpha")
elif args.ransac and args.min_sample is None:
    parser.error("--ransac requires --min_sample")
elif args.nlr and args.max_depth is None:
    parser.error("--nlr requires --max_depth")

if __name__ == "__main__":
    if args.lr:
        regressor = MyLinearRegression()
    elif args.lrne:
        regressor = MyLinearRegressionNE()
    elif args.lasso:
        regressor = MyLasso(alpha=args.alpha)
    elif args.ridge:
        regressor = MyRidge(alpha=args.alpha)
    elif args.ransac:
        regressor = MyRANSACRegressor(min_sample=args.min_sample)
    elif args.nlr:
        regressor = MyDecisionTreeRegressor(max_depth=args.max_depth)

# load data set
if path.exists(args.dataSet):
    if args.dataSet == "../csv/housing.data.txt":
        df = pd.read_csv(args.dataSet, header=None, delim_whitespace=True)
    else:
        df = pd.read_csv(args.dataSet, header=None)
    # select labels
    Y = df.iloc[:, -1].values.reshape(-1, 1)
    # extract features
    X = df.iloc[:, 0:-1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
else:
    raise ValueError("Data set not found !")

# Standardize the features
sc_x = StandardScaler()
X_train_std = sc_x.fit_transform(X_train)
X_test_std = sc_x.transform(X_test)
sc_y = StandardScaler()
Y_train_std = sc_y.fit_transform(Y_train)
Y_test_std = sc_y.transform(Y_test)
# Start the experiment
print("REGRESSOR: " + regressor.name)
# Start to measure running time of training process
start_time = time.time()
# fit the model
regressor.fit(X_train_std, Y_train_std)
print("Time for training is  %s seconds" % (time.time() - start_time))
# predict
Y_train_pred_std = regressor.predict(X_train_std)
Y_test_pred_std = regressor.predict(X_test_std)
# Get the predicted value in original space
# Y_train_pred = sc_y.inverse_transform(Y_train_pred_std)
# Y_test_pred = sc_y.inverse_transform(Y_test_pred_std)
# performance metrics
error_train = mean_squared_error(Y_train_std, Y_train_pred_std)
error_test = mean_squared_error(Y_test_std, Y_test_pred_std)
print('MSE train: %.3f, test:%.3f' % (error_train, error_test))
r2_train = r2_score(Y_train_std, Y_train_pred_std)
r2_test = r2_score(Y_test_std, Y_test_pred_std)
print('R^2 train: %.3f, test:%.3f' % (r2_train, r2_test))
