#  Copyright (c) 2020. Tuan-Hiep TRAN
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import sys

# download mnist dataset
mnist = fetch_openml('mnist_784')
# create dataframe for 1000 samples from mnist dataset
x = mnist.data[:1000, :]
y = mnist.target[:1000]
X = pd.DataFrame(x)
Y = pd.DataFrame(y)
df = pd.concat([X, Y], axis=1)
# save this subset to .csv file
np.savetxt(sys.argv[1], df, delimiter=',', fmt='%s')
