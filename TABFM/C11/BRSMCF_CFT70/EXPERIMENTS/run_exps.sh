#!/bin/bash

nfolds=3

for f in $(ls DATA)
do
	output_file=$(basename $f)
	python c11_regression_kfold.py DATA/$f RESULTS/$output_file $nfolds
done
