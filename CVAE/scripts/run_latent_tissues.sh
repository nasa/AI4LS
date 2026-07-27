#!/bin/bash

python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/latent_gene_predictor.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/latent_gene_pred/by_tissue/ \
    --by_tissue \
    --min_samples 5 \
    --validate
