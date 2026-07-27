#!/bin/bash

python /home/jcasalet/nobackup/AI4LS/CVAE/src/visualize_latent.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue_v3/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/visualize/tissue_v3/
