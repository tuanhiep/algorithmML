#  Copyright (c) 2020. Tuan-Hiep TRAN

from sklearn import metrics

labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [-1, -1, 3, 3, 2, 2]

score= metrics.adjusted_rand_score(labels_true, labels_pred)
print(score)