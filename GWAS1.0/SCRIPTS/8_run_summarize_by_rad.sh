#!/bin/bash -x

RESULTS_DIR=/home/jcasalet/nobackup/GWAS/DATA/RESULTS
python summarize_by_rad.py $RESULTS_DIR/snp-summary-per-chrom.json > $RESULTS_DIR/snp-summary-per-rad.json
