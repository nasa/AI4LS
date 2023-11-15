#!/bin/bash -x

for i in $(seq 1 22)
do
	sbatch /home/jcasalet/nobackup/GWAS/DATA/SCRIPTS/do_gwas.sh /home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/$i/FILTER_BEFORE_MERGE/MERGED/FILTER_AFTER_MERGE/filteraftermerge.vcf
done
