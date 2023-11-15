#!/bin/bash -x

for i in $(seq 1 22)
do
	sbatch -t 01-00:00:00 /home/jcasalet/nobackup/GWAS/DATA/SCRIPTS/filter_after_merge.sh /home/jcasalet/nobackup/GWAS/DATA/VCF/OUT/$i/FILTER_BEFORE_MERGE/MERGED/
done
