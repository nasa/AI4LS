#!/bin/bash -x

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3066182/
ANNOTATION_FILE=/home/jcasalet/nobackup/GWAS/DATA/Homo_sapiens.GRCh37.87.bed.gz

BCFTOOLS_111=/home/jcasalet/nobackup/GWAS/BCFTOOLS/bcftools-1.11/bcftools
BCFTOOLS_118=/home/jcasalet/nobackup/GWAS/BCFTOOLS/bcftools-1.18/bcftools

MISSING_RATE=0.10

if [ $# -ne 1 ]
then
	echo "usage: $0 <input_dir>"
	exit 1
fi

IN_DIR=$1
TS=$(date +%s)

if [ -z "$IN_DIR" -o ! -d "$IN_DIR" ]
then
	echo "usage: $0 <input_dir>"
	exit 1
fi

if [ -d ${IN_DIR}/FILTER_BEFORE_MERGE ]
then
	mv ${IN_DIR}/FILTER_BEFORE_MERGE ${IN_DIR}/FILTER_BEFORE_MERGE-$TS
fi

mkdir ${IN_DIR}/FILTER_BEFORE_MERGE

for f in $(ls ${IN_DIR}/*.vcf)
do
	bgzip $f
	tabix ${f}.gz
done

for f in $(ls ${IN_DIR}/*.vcf.gz)
do
	out_filename=$(echo $f | cut -d. -f1)
	out_filename=$(basename $out_filename)
	out_filename=${IN_DIR}/FILTER_BEFORE_MERGE/${out_filename}

	# remove multi-allelic vars
	$BCFTOOLS_118 view --max-alleles 2 ${f} > ${out_filename}_1.vcf 
	#plink2 --max-alleles 2 --vcf ${f} $f --out ${out_filename}_1 --export vcf 

	# filter out LOWCONF
	$BCFTOOLS_118 view -f PASS ${out_filename}_1.vcf > ${out_filename}_2.vcf 
	#plink2 --var-filter --vcf ${out_filename}_1.vcf --out ${out_filename}_2 --export vcf

	# include only SNPs (filter out INDELs)
        $BFTOOLS_118 view --types snps ${out_filename}_2.vcf > ${out_filename}_3.vcf 

	# expand multi-allelic sites
	#bcftools norm --multiallelics '-both' ${f} --output ${f} 
	# rename variants by coordinates
	#bcftools annotate --set-id '%CHROM\_%POS\_%REF\_%ALT'  ${f} --output ${f} 

	# remove samples with sex inconsistency b/w X/Y chroms and reported sex

	# remove all but one in IBD?

	# rename final output file
	mv ${out_filename}_3.vcf ${out_filename}-filterbeforemerge.vcf
	
	# remove intermediate files
	rm -f ${out_filename}_*.vcf

done
