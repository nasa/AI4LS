#!/bin/bash


python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/check_auroc.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue_v3_redo/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/check_auroc/finetune/tissue_v3_redo/

python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/inference.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue_v3_redo/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/inference/finetune/tissue_v3_redo/ \
    --skip_enrichment

python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/check_latent_dims.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue_v3_redo/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/check_latent_dims/finetune/tissue_v3_redo/

python /home/jcasalet/nobackup/AI4LS/CVAE/src/visualize_latent.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue_v3_redo/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/visualize/tissue_v3_redo/
