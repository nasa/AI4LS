#!/bin/bash -x

OUT_DIR=/home/jcasalet/nobackup/GWAS/DATA/RESULTS

python summarize_by_chrom.py | python -m json.tool  > $OUT_DIR/snp-summary-per-chrom.json
