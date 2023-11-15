#!/bin/bash -x

for i in $(seq 1 22)
do
	sbatch -t 03-00:00:00 /home/jcasalet/nobackup/GWAS/DATA/SCRIPTS/merge_vcf.sh /home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/$i/FILTER_BEFORE_MERGE
done
