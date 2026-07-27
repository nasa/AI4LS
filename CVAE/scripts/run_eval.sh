#!/bin/bash

python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/inference.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/inference/finetune/tissue/ \
    --skip_enrichment
