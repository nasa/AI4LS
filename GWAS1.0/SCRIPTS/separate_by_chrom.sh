#!/bin/bash -x

VCF_DIR=$1
START_CHROM=$2
STOP_CHROM=$3

if [ $# -ne 3 ]
then
	echo "usage: $0 <vcf_dir> <start-chrom> <end-chrom>"
	exit 1
fi

if [ ! -d "$VCF_DIR" ]
then
	echo "usage: $0 <vcf_dir> <start-chrom> <end-chrom>"
	exit 1
fi

if [ "$START_CHROM" == "X" -o "$START_CHROM" == "x" ]
then
	mkdir -p ${VCF_DIR}/OUT/X
	for f in $(ls ${VCF_DIR}/*.vcf.gz)
        do
                file_name=$(basename $f)
                file_name=$(echo $file_name | cut -d. -f1)
                #bcftools filter -i"chrom='$i'" $f --output ${VCF_DIR}/OUT/${i}/${file_name}
                plink2 --chr X --vcf $f --out ${VCF_DIR}/OUT/X/${file_name} --export vcf --update-sex /home/jcasalet/nobackup/GWAS/DATA/PHENO_COVAR/covar.tsv
        done

else
	for i in $(seq $START_CHROM $STOP_CHROM)
	do
		if [ ! -d ${VCF_DIR}/OUT/$i ]
        	then
                	mkdir -p ${VCF_DIR}/OUT/$i
        	fi
		for f in $(ls ${VCF_DIR}/*.vcf.gz)
		do
			file_name=$(basename $f)
			file_name=$(echo $file_name | cut -d. -f1)
			#bcftools filter -i"chrom='$i'" $f --output ${VCF_DIR}/OUT/${i}/${file_name}
			plink2 --chr $i --vcf $f --out ${VCF_DIR}/OUT/${i}/${file_name} --export vcf
		done
	done
fi
