#  Copyright (c) 2020. Tuan-Hiep TRAN

import argparse
import A5

# parse the arguments from command line
parser = argparse.ArgumentParser(description="Dimensionality Reduction")
parser.add_argument("-n", "--numberComponent", type=int, required=True, help="number of components for this analysis")
parser.add_argument("-data", "--dataSet", type=str, required=True, help="name of the data set for this program")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-pca", "--pca", action="store_true", help="use PCA ")
group.add_argument("-lda", "--lda", action="store_true", help="use LDA ")
group.add_argument("-kpca", "--kpca", action="store_true", help="use KPCA ")
parser.add_argument("-k", "--kernel", type=str, help="kernel method")
parser.add_argument("-g", "--gamma", type=int, help="gamma parameter")
args = parser.parse_args()

if args.kpca and (args.kernel is None or args.gamma is None):
    parser.error("--kpca requires --kernel and --gamma")


if __name__ == "__main__":
    if args.pca:
        pass
    elif args.lda:
        pass
    elif args.kpca:
        pass

if args.dataSet == "iris":
    pass
else:
    pass


# Standardize the features

# Realize the dimensionality reduction

# Get the result


# PCA
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4,
random_state=1)
tree_model.fit(X_train_lda, y_train)
X_test_pca = pca.transform(X_test_std)
y_pred = tree_model.predict(X_test_pca)
acc = accuracy_score(y_pred, y_test)
print("DT+PCA acc=", acc)

# LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4,
random_state=1)
tree_model.fit(X_train_lda, y_train)
X_test_lda = lda.transform(X_test_std)
y_pred = tree_model.predict(X_test_lda)
acc = accuracy_score(y_pred, y_test)
print("DT+LDA acc=", acc)

# KPCA

from sklearn.decomposition import KernelPCA
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.transform(X_test)
ppn1 = Perceptron(eta0=0.1, random_state=1)
ppn1.fit(X_train, y_train)
y_pred1 = ppn.predict(X_test)
print("accuracy (no kpca) = ", sum(y_test==y_pred1)/y_test.shape[0])
ppn2 = Perceptron(eta0=0.1, random_state=1)
ppn2.fit(X_train_kpca, y_train)
y_pred2 = ppn.predict(X_test_kpca)
print("accuracy (with kpca) ", sum(y_test==y_pred2)/y_test.shape[0])