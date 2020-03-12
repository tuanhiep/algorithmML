#  Copyright (c) 2020. Tuan-Hiep TRAN
import numpy as np

# initial values
weights = np.array([0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.167, 0.167, 0.167, 0.072])
label_y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
predicted_y = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, -1])

# For each boosting round of AdaBoost algorithm

# 2(c) Compute weighted error rate epsilon

epsilon = np.dot(weights, (label_y != predicted_y).astype(int))

# 2(d) Compute coefficient

coefficient = 0.5 * np.log((1 - epsilon) / epsilon)

# 2(e) Update weights

weights = weights * np.exp(-epsilon * predicted_y * label_y)

# 2(f) Normalize weights to sum to 1

weights = weights / weights.sum()

# print out the weights for next round

print(weights)

