#!/bin/bash -x

if [ $# -ne 2 ]
then
	echo "usage: $0 <chrom> <offset>"
	exit 1
fi


CHROM=$1
DIR_NAME=/home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/${CHROM}/FILTER_BEFORE_MERGE/MERGED/FILTER_AFTER_MERGE
VCF_FILE=filteraftermerge.vcf
OUT_FILE=/home/jcasalet/nobackup/GWAS/DATA/RESULTS/gene-summary-per-chrom-${CHROM}.txt
OFFSET=$2

cd $DIR_NAME
if [ ! -f ${VCF_FILE}.gz ]
then
	bgzip $VCF_FILE
	tabix ${VCF_FILE}.gz
fi

if [ -f ${VCF_FILE}.gz -a ! -f ${VCF_FILE}.gz.tbi ]
then
	tabix ${VCF_FILE}.gz
fi

ret=$(python /home/jcasalet/nobackup/GWAS/DATA/SCRIPTS/summarize_by_gene.py /home/jcasalet/nobackup/GWAS/DATA/RESULTS/snp-summary-per-chrom.json $CHROM $OFFSET)

echo $ret > $OUT_FILE

