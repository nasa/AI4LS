#!/bin/bash
python /home/jcasalet/nobackup/CVAE/src/inference.py \
    --checkpoint /home/jcasalet/nobackup/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/CVAE/DATA/subset_final.h6 \
    --output_dir /home/jcasalet/nobackup/CVAE/results/inference/v9/ \
    --skip_enrichment False \
    --enrichment_genes 100 \
    --enrichment_cutoff 0.05
