#!/bin/bash

for f in $(ls DATA)
do
	output_file=$(basename $f)
	python c11_regression.py DATA/$f RESULTS/$output_file
done
