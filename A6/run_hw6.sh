#!/bin/bash
#Start the experiments
echo "Starting..."

# Prepare second datasets
echo "Preparing second datasets..."
python -W ignore prepare_dataset.py csv/all_breakdown.csv csv/all_breakdown_fine.csv
echo "DOING EXPERIMENTS:"
#HOUSING DATA SET
echo "HOUSING DATA SET"
dataset="csv/housing.data.txt"
echo "Do experiment with $dataset data set: "
for regressor in "lr" "lrne"; do
  python -W ignore main.py -data $dataset "-${regressor}"
done

for alpha in  0.1 1 10; do
  echo "ALPHA = ${alpha}"
  python -W ignore main.py -data $dataset -lasso -alpha ${alpha}
  python -W ignore main.py -data $dataset -ridge -alpha ${alpha}
done

for min_sample in 50 150 250 ; do
  echo "MIN SAMPLE = ${min_sample}"
  python -W ignore main.py -data $dataset -ransac -min_sample ${min_sample}
done

for max_depth in  5 9 13; do
  echo "MAX DEPTH = ${max_depth}"
  python -W ignore main.py -data $dataset -nlr -max_depth ${max_depth}
done

#ALL BREAKDOWN DATA SET
echo "ALL BREAKDOWN DATA SET"
dataset="csv/all_breakdown_fine.csv"
echo "Do experiment with $dataset data set: "
for regressor in "lr" "lrne"; do
  python -W ignore main.py -data $dataset "-${regressor}"
done

for alpha in  0.1 1 10; do
  echo "ALPHA = ${alpha}"
  python -W ignore main.py -data $dataset -lasso -alpha ${alpha}
  python -W ignore main.py -data $dataset -ridge -alpha ${alpha}
done

for min_sample in 50 150 250 ; do
  echo "MIN SAMPLE = ${min_sample}"
  python -W ignore main.py -data $dataset -ransac -min_sample ${min_sample}
done

for max_depth in 3 5 7; do
  echo "MAX DEPTH = ${max_depth}"
  python -W ignore main.py -data $dataset -nlr -max_depth ${max_depth}
done

echo "End !"
