#!/bin/bash

IN_DIR=$1
if [ ! -d "$IN_DIR" ]
then
	echo "directory $IN_DIR doesn't exist"
	exit 1
fi

if [ -d "$IN_DIR"/SIG_FDR_BH ]
then
	TS=$(date +%s)
	mv "$IN_DIR"/SIG_FDR_BH "$IN_DIR"/SIG_FDR_BH-$TS
fi

python analyze_gwas.py $IN_DIR 


