#!/bin/bash

# by index
#python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/knn_latent.py \
#    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v2/best_model.pt \
#    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
#    --mode sample \
#    --sample_idx 1968 \
#    --k 10 \
#    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/knn/sample_1968

# by metadata — finds mean z of all matching samples
python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/knn_latent.py \
    --checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v2/best_model.pt \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    --mode sample \
    --tissue Retina \
    --k 20 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/results/knn/retina
    #--tissue Adipose --strain C57BL/6J --flight 0 --sex Male --euth Isoflurane\

# from a file
#python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/knn_latent.py \
    #--checkpoint /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/finetune/v2/best_model.pt \
    #--data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/osdr_mouse.h5 \
    #--mode file \
    #--query_file /home/jcasalet/nobackup/AI4LS/CVAE/my_sample.csv \
    #--query_tissue Liver \
    #--query_strain C57BL/6J \
    #--query_sex Female \
    #--k 10
