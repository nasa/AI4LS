#!/bin/bash

# specific tissue
python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/decoder_jacobian.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/jacobian_results/bone_marrow/ \
    --tissue "Bone Marrow"

# all tissues
python -u  /home/jcasalet/nobackup/AI4LS/CVAE/decoder_jacobian.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/jacobian_results/by_tissue/ \
    --by_tissue --min_samples 5

# global
python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/decoder_jacobian.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/tissue/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/jacobian_results/global/


