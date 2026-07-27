#!/bin/bash

python -u  /home/jcasalet/nobackup/AI4LS/CVAE/src/check_study_pred.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/metadata/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse_filtered.h5
