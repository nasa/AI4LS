#!/bin/bash -x

# https://www.biostars.org/p/392994/

BCFTOOLS=/home/jcasalet/nobackup/GWAS/BCFTOOLS/bcftools-1.11/bcftools

if [ $# -ne 1 ]
then
	echo "usage: $0 <in-dir>"
	exit 1
fi

IN_DIR=$1

if [ ! -d $IN_DIR ]
then
	echo directory $IN_DIR does not exist
	exit 1
fi

if [ -d ${IN_DIR}/MERGED ]
then
	TS=$(date +%s)
	mv ${IN_DIR}/MERGED ${IN_DIR}/MERGED-$TS
fi

mkdir ${IN_DIR}/MERGED

for f in $(ls ${IN_DIR}/*.vcf)
do
	bgzip $f
	tabix ${f}.gz
done

FILE_LIST=$(ls ${IN_DIR}/*.vcf.gz | xargs) 
$BCFTOOLS merge $FILE_LIST --output ${IN_DIR}/MERGED/merged.vcf

