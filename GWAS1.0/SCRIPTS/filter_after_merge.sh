#!/bin/bash -x

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3066182/
export BCFTOOLS_PLUGINS=/home/jcasalet/nobackup/GWAS/BCFTOOLS/bcftools-1.18/plugins/
BCFTOOLS_118=/home/jcasalet/nobackup/GWAS/BCFTOOLS/bcftools-1.18/bcftools
ANNOTATION_FILE=/home/jcasalet/nobackup/GWAS/DATA/Homo_sapiens.GRCh37.87.bed.gz

if [ $# -ne 1 -o ! -d "$1" -o ] 
then
	echo "usage: $0 <in-vcf-dir>"
	exit 1
fi

IN_DIR=$1

if [ ! -f ${IN_DIR}/merged.vcf ]
then
	echo "${IN_DIR}/merged.vcf doesn't exist"
	exit 1
fi

OUT_DIR=${IN_DIR}/FILTER_AFTER_MERGE


if [ -d $OUT_DIR ]
then
	TS=$(date +%s)
	mv $OUT_DIR ${OUT_DIR}_$TS
fi

mkdir $OUT_DIR

out_filename="filteraftermerge-temp"

# remove multi-allelic (again!)
$BCFTOOLS_118 view --max-alleles 2 ${IN_DIR}/merged.vcf > ${OUT_DIR}/${out_filename}_1.vcf

# annotate with gene info
$BCFTOOLS_118 annotate -a $ANNOTATION_FILE -c CHROM,FROM,TO,GENE  -h <(echo '##INFO=<ID=GENE,Number=1,Type=String,Description="Gene name">') ${OUT_DIR}/${out_filename}_1.vcf > ${OUT_DIR}/${out_filename}_2.vcf 

# annotate with biotype
$BCFTOOLS_118 annotate -a $ANNOTATION_FILE -c biotype -h <(echo '##INFO=<biotype=protein_coding,Number=1,Type=String,Description="Gene name">') ${OUT_DIR}/${out_filename}_1.vcf > ${OUT_DIR}/${out_filename}_2.vcf 

# exclude non-protein-coding genes
#$BCFTOOLS_118 filter -e'INFO/GENE="."' ${OUT_DIR}/${out_filename}_2.vcf > ${OUT_DIR}/${out_filename}_3.vcf 
$BCFTOOLS_118 filter -i'INFO/biotype=protein_coding' ${OUT_DIR}/${out_filename}_2.vcf > ${OUT_DIR}/${out_filename}_3.vcf 

# include only SNPs
$BCFTOOLS_118 view --types snps ${OUT_DIR}/${out_filename}_3.vcf > ${OUT_DIR}/${out_filename}_4.vcf 

# annotate with AF info
$BCFTOOLS_118 +fill-tags ${OUT_DIR}/${out_filename}_4.vcf -- -t AF > ${OUT_DIR}/${out_filename}_5.vcf

# filter MAF < 0.05
$BCFTOOLS_118 view -i'MAF>0.05' ${OUT_DIR}/${out_filename}_5.vcf > ${OUT_DIR}/${out_filename}_6.vcf

# remove if not hwe
plink2 --hwe 0.05 --vcf ${OUT_DIR}/${out_filename}_6.vcf --export vcf --out ${OUT_DIR}/${out_filename}_7

# remove missingness
# $BCFTOOLS_118 view -i'F_MISSING < ${MISSING_RATE}' ${out_filename}_3.vcf > ${out_filename}_4.vcf
plink2 --geno 0.10 --vcf ${OUT_DIR}/${out_filename}_7.vcf --export vcf --out ${OUT_DIR}/${out_filename}_8
plink2 --mind 0.10 --vcf ${OUT_DIR}/${out_filename}_8.vcf --export vcf --out ${OUT_DIR}/${out_filename}_9

# rename final file
mv ${OUT_DIR}/${out_filename}_9.vcf ${OUT_DIR}/filteraftermerge.vcf

# remove intermediate files
rm -f ${OUT_DIR}/${out_filename}_*



# check gender
#plink2 --check-sex ${IN_DIR}/${f} --out ${IN_DIR}/${f}

# run pca to see if population structure
#plink2 --pca var-wts ${IN_DIR}/${f} --out ${IN_DIR}/${f}

# check for cryptic relatedness (correct for population structure)
#plink2 --genome --min 0.20 ${IN_DIR}/${f} --out ${IN_DIR}/${f}

# run pca to see if population structure
#plink2 --pca var-wts ${IN_DIR}/${f} --out ${IN_DIR}/${f}

# imputation quality
#plink2 --mach-r2-filter <min> <max>

# remove if not HWE?
#plink2 --hwe 0.05 ${IN_DIR}/${f} --out ${
#plink2 --hardy

# move final file names
#mv ${OUT_DIR}/${out_filename}_2.pgen ${OUT_DIR}/${out_filename}-filteraftermerge.pgen
#mv ${OUT_DIR}/${out_filename}_2.bim ${OUT_DIR}/${out_filename}-filteraftermerge.bim
#mv ${OUT_DIR}/${out_filename}_2.fam ${OUT_DIR}/${out_filename}-filteraftermerge.fam

# remove intermediate files
#rm ${OUT_DIR}/${out_filename}_.*
