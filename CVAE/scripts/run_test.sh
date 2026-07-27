#!/bin/bash

# test set metrics
#python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/inference.py \
    #--checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v4/best_model.pt \
    #--data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/subset_final.h6 \
    #--output_dir results/inference/finetune/v4/ \
    #--skip_enrichment

# dimension variance
#python -u  /home/jcasalet/nobackup/AI4LS/CVAE/src/check_latent_dims.py \
    #--checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v4/best_model.pt

# within-study AUROC
#python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/check_auroc.py \
    #--checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v4/best_model.pt \
    #--data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/subset_final.h6

#python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/inference.py \
#    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v5/best_model.pt \
#    --data /home/jcasalet/nobackup/AI4LS/CVAE//DATA/subset_final.h6 \
#    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/inference/finetune/v5/ \
#    --skip_enrichment

# test set metrics
python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/inference.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/inference/finetune/tissue/

# dimension variance
python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/check_latent_dims.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 

# within-study AUROC
python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/check_auroc.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 
