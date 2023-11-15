#!/bin/bash

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3066182/
export BCFTOOLS_PLUGINS=/home/jcasalet/nobackup/GWAS/BCFTOOLS/bcftools-1.18/plugins/

if [ $# -ne 1 -a -z "$IN_DIR" ]
then
	echo "usage: $0 <input_dir>"
	exit 1
fi

IN_DIR=$1
OUT_DIR=${1}/FINAL_FILTER

for f in $(ls ${IN_DIR})
do
	file_basename=$(basename $f)
	file_basename=$(echo $file_basename | cut -d. -f1)		

	# annotate with AF info
	bcftools +fill-tags ${IN_DIR}/${f} -- -t AF > $OUT_DIR/${file_basename}_1.vcf

	# filter MAF < 0.05
	bcftools view -i'MAF>0.05' ${OUT_DIR}/${file_basename}_1.vcf > ${OUT_DIR}/${file_basename}_2.vcf 
	#plink2 --maf 0.05 ${IN_DIR}/${f} --out ${IN_DIR}/${f} 

	# run pca to see if population structure
	plink2 --pca var-wts ${IN_DIR}/${f} --out ${IN_DIR}/${f}

	# check for cryptic relatedness (correct for population structure)
	plink2 --genome --min 0.20 ${IN_DIR}/${f} --out ${IN_DIR}/${f}

	# run pca to see if population structure
	plink2 --pca var-wts ${IN_DIR}/${f} --out ${IN_DIR}/${f}
	
        # annotate with gene info
        bcftools annotate -a $ANNOTATION_FILE -c CHROM,FROM,TO,GENE  -h <(echo '##INFO=<ID=GENE,Number=1,Type=String,Description="Gene name">') $f --output ${f}

        # exclude non-protein-coding genes
        bcftools filter -e'INFO/GENE="."' ${f} --output ${f}

	# include only SNPs
        bcftools view --types snps ${f} --output ${f}


done
