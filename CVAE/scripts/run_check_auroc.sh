#!/bin/bash

python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/check_auroc.py --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue/best_model.pt --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 

# tissue only — overrides whatever the checkpoint was trained with
#python check_auroc.py \
    #--checkpoint checkpoints/finetune/tissue/best_model.pt \
    #--data DATA/osdr_mouse.h5 \
    #--conditions tissue

# tissue + strain
#python check_auroc.py \
    #--checkpoint checkpoints/finetune/v2/best_model.pt \
    #--data DATA/osdr_mouse.h5 \
    #--conditions tissue strain

# default — uses checkpoint conditions
#python check_auroc.py \
    #--checkpoint checkpoints/finetune/v2/best_model.pt \
    #--data DATA/osdr_mouse.h5
