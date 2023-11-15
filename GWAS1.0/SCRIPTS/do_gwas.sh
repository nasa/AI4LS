#!/bin/bash -x

# https://www.cog-genomics.org/plink/2.0/assoc

# ./plink --vcf cleaned.recode.vcf --pheno phenotype_last.txt --allow-no-sex -no-parents --vcf-half-call haploid --allow-extra-chr --chr-set -1 --assoc fisher

if [ $# -ne 1 -a -f "$1" ]
then
	echo "usage: $0 <in-vcf>"
	exit 1
fi

VCF_FILE=$1

out_filename=$(echo $VCF_FILE | cut -d. -f1)
out_filename=$(basename $out_filename)

OUT_DIR=$(dirname $VCF_FILE)/GWAS

if [ -d $OUT_DIR ]
then
	TS=$(date +%s)
	mv $OUT_DIR ${OUT_DIR}_$TS
fi

mkdir $OUT_DIR

cd $(dirname $VCF_FILE)
#bgzip $VCF_FILE
#tabix $VCF_FILE
#plink2 --adjust --allow-no-sex --ci 0.95 --covar pca.eigenvec --covar-number 1 --vcf bob --logistic --out joan
#plink2 --allow-no-covars --adjust --allow-no-sex --ci 0.95 --vcf $VCF_FILE --logistic --out ${OUT_DIR}/out_filename 
plink2 --adjust --allow-no-sex --ci 0.95 --vcf ${VCF_FILE}.gz --glm --pheno /home/jcasalet/nobackup/GWAS/DATA/PHENO_COVAR/pheno.tsv --covar /home/jcasalet/nobackup/GWAS/DATA/PHENO_COVAR/covar.tsv  --covar-variance-standardize --out ${OUT_DIR}/out_filename 


