#!/bin/bash -x

RESULTS_DIR=/home/jcasalet/nobackup/GWAS/DATA/RESULTS

for i in $(seq 1 22)
do
	sbatch /home/jcasalet/nobackup/GWAS/DATA/SCRIPTS/summarize_by_gene.sh $i 0 
done	
