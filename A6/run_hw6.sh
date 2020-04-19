#!/bin/bash
#Start the experiments
echo "Starting..."

# Prepare second datasets
echo "Preparing second datasets..."
python -W ignore prepare_dataset.py csv/all_breakdown.csv csv/all_breakdown-clean.csv

#HOUSING DATA SET
dataset="../csv/housing.data.txt"
echo "Do experiment with $dataset data set: "
for regressor in "lr" "lrne"; do
  python -W ignore main.py -data $dataset "-${regressor}"
done

python -W ignore main.py -data $dataset -lasso -alpha 1.0
python -W ignore main.py -data $dataset -ridge -alpha 1.0
python -W ignore main.py -data $dataset -ransac -min_sample 50
python -W ignore main.py -data $dataset -nlr -max_depth 13

#ALL BREAKDOWN DATA SET
echo "ALL BREAKDOWN DATA SET"
dataset="../csv/all_breakdown_fine.csv"
echo "Do experiment with $dataset data set: "
for regressor in "lr" "lrne"; do
  python -W ignore main.py -data $dataset "-${regressor}"
done

python -W ignore main.py -data $dataset -lasso -alpha 1.0
python -W ignore main.py -data $dataset -ridge -alpha 1.0
python -W ignore main.py -data $dataset -ransac -min_sample 50
python -W ignore main.py -data $dataset -nlr -max_depth 13

echo "End !"