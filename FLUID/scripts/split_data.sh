#!/bin/bash

if [ $# -ne 3 ]
then
	echo usage: $0 data-csv-file COLAB_SPLIT TEST_SPLIT
	echo "example: $0 data.csv .5 .2"
	exit 1
fi

python split_data.py $1 $2 $3

if [ $? -ne 0 ]
then
	echo "error running $0"
	exit 1
fi


mkdir -p ../data/col_0/test
mkdir -p ../data/col_1/test
mkdir -p ../data/col_0/train
mkdir -p ../data/col_1/train

mv colab0_test.csv ../data/col_0/test/data.csv
mv colab1_test.csv ../data/col_1/test/data.csv
mv colab0_train.csv ../data/col_0/train/data.csv
mv colab1_train.csv ../data/col_1/train/data.csv


for d in COLAB_EARTH COLAB_ISS COLAB_SHIM AGG_EARTH AGG_ISS
do
	cp -r ../data/col_0 ../data/WORKSPACE/$d
	cp -r ../data/col_1 ../data/WORKSPACE/$d
done
