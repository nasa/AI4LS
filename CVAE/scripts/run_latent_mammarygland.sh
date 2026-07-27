#!/bin/bash

python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/latent_gene_predictor.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue_64_32/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/latent_gene_pred/tissue_64_32/by_tissue/Mammary_Gland/ \
    --tissue "Mammary Gland" \
    --validate
