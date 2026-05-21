#!/bin/bash

python /home/jcasalet/nobackup/CVAE/src/visualize_latent.py \
    --checkpoint /home/jcasalet/nobackup/CVAE/checkpoints/v9/best_model.pt \
    --data /home/jcasalet/nobackup/CVAE/DATA/subset_final.h6 \
    --output_dir /home/jcasalet/nobackup/CVAE/results/visualize/v9
