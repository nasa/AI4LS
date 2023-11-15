#!/bin/bash

BCFTOOLS=/home/jcasalet/nobackup/GWAS/BCFTOOLS/bcftools-1.11/bcftools

if [ $# -ne 1 ]
then
        echo "usage: $0 <out-dir>"
        exit 1
fi

FILE_LIST=""

OUT_DIR=$1

for i in $(seq 1 22)
do
	FILE_LIST="$FILE_LIST /home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/$i/FILTER_BEFORE_MERGE/MERGED/FILTER_AFTER_MERGE/filteraftermerge.vcf.gz"
done

$BCFTOOLS concat $FILE_LIST --output ${OUT_DIR}/all_merged.vcf
