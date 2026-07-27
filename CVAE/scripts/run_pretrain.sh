#!/bin/bash
#python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/pretrain_archs4_zeropad.py \
#python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/train.py \
    #--data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/archs4_pretrain_no_unknown_no_other.h5 \
    #--output_dir /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/pretrain/tissue_v3 \
    #--conditions tissue \
    #--epochs 500 \
    #--batch_size 128 \
    #--patience 40 \
    #--latent_dim 64

#python /home/jcasalet/nobackup/AI4LS/CVAE/src/pretrain_archs4.py \
    #--data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/archs4_pretrain_no_unknown_no_other_subset.h5 \
    #--output_dir /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/pretrain/tissue_8_study_8_latent_32 \
    #--conditions tissue study \
    #--latent_dim 32 \
    #--tissue_emb_dim 8 \
    #--study_emb_dim 8 \
    #--epochs 500 \
    #--batch_size 128 \
    #--patience 40 \
    #--grl_alpha 2.0


python -u /home/jcasalet/nobackup/AI4LS/CVAE/src/pretrain_archs4.py  \
    --data /home/jcasalet/nobackup/AI4LS/CVAE/DATA/archs4_pretrain.h5 \
    --output_dir /home/jcasalet/nobackup/AI4LS/CVAE/checkpoints/pretrain/tissue_64_32 \
    --epochs 500 \
    --conditions tissue \
    --latent_dim 64 \
    --tissue_emb_dim 32 \
    --epochs 500 \
    --kl_anneal_epochs 100 \
    --patience 80 \
    --batch_size 128 \
    --beta 0.01 \
    --nsamples 1000

