#  Copyright (c) 2020. Tuan-Hiep TRAN

import argparse
import A5

# parse the arguments from command line
parser = argparse.ArgumentParser(description="Dimensionality reduction")
parser.add_argument("-n", "--numberComponent", type=int, required=True, help="Number of components for this analysis")
parser.add_argument("-data", "--dataSet", type=str, required=True, help="Name of the data set for this program")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-pca", "--pca", action="store_true", help="use PCA ")
group.add_argument("-lda", "--lda", action="store_true", help="use LDA ")
group.add_argument("-kpca", "--kpca", action="store_true", help="use KPCA ")
args = parser.parse_args()

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
