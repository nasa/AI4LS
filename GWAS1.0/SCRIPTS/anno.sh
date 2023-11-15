#!/bin/bash

BCFTOOLS_118=/home/jcasalet/nobackup/GWAS/BCFTOOLS/bcftools-1.18/bcftools
#ANNOTATION_FILE=/home/jcasalet/nobackup/GWAS/DATA/Homo_sapiens.GRCh37.87.bed.gz
#ANNOTATION_FILE=/home/jcasalet/nobackup/GWAS/DATA/gencode.v37lift37.annotation.gtf 
#ANNOTATION_FILE=/home/jcasalet/nobackup/GWAS/DATA/Homo_sapiens.GRCh37.87.gff3
ANNOTATION_FILE=/home/jcasalet/nobackup/GWAS/DATA/anno_test.bed
#$BCFTOOLS_118 annotate -a $ANNOTATION_FILE -c CHROM,FROM,TO,INFO/GENE,INFO/biotype  -h header-bed.txt ../merged.vcf > gene.vcf  

#$BCFTOOLS_118 annotate -a $ANNOTATION_FILE -c CHROM,FROM,TO,ID,INFO/GENE,INFO/biotype  -h header-bed.txt ../merged.vcf > gene.vcf  

#$BCFTOOLS_118 annotate -a $ANNOTATION_FILE -c CHROM,FROM,TO,GENE,biotype -h <(echo '##INFO=<ID=biotype,Number=1,Type=String,Description="Protein-coding gene">') ../merged.vcf > biotype.vcf  
#$BCFTOOLS_118 annotate -a $ANNOTATION_FILE -c INFO ../VCF/OUT/22/FILTER_BEFORE_MERGE/MERGED/merged.vcf > biotype.vcf  

$BCFTOOLS_118 annotate -a $ANNOTATION_FILE -c CHROM,FROM,TO,INFO/TRANSCRIPT -h header-bed.txt ../test.vcf > biotype.vcf  
#$BCFTOOLS_118 annotate -a $ANNOTATION_FILE -c CHROM,FROM,TO,GENE,biotype -h header.txt ../merged.vcf > biotype.vcf  
